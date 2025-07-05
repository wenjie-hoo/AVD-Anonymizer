import bisect

import cv2
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torchvision
from mmengine import Config
from mmengine.dataset import Compose, pseudo_collate
from mmengine.device import get_device

try:
    from pytorch_grad_cam import AblationCAM, AblationLayer, EigenCAM
    from pytorch_grad_cam.utils.image import scale_cam_image, show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam"` to install '
                      '3rd party package pytorch_grad_cam.')

from mmdet.apis import init_detector, inference_detector
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample


def reshape_transform(activations, *args, **kwargs):
    """
    Fixed reshape_transform function that accepts any additional arguments
    from pytorch_grad_cam library.
    """
    print(f"Input type: {type(activations)}")
    if isinstance(activations, (list, tuple)):
        print(f"List/tuple length: {len(activations)}")
        for i, act in enumerate(activations):
            if hasattr(act, 'shape'):
                print(f"  Item {i} shape: {act.shape}")
        
        results = []
        for activation in activations:
            # Ensure we have 4D tensor (B, C, H, W)
            if activation.dim() == 3:
                activation = activation.unsqueeze(0)
            
            if len(results) == 0:
                target_size = activation.shape[-2:]
                results.append(activation)
            else:
                resized = torch.nn.functional.interpolate(
                    activation, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
                results.append(resized)
        
        return torch.cat(results, dim=1)
    
    elif hasattr(activations, 'shape'):
        print(f"Tensor shape: {activations.shape}")
        # Single tensor case
        if activations.dim() == 3:
            activations = activations.unsqueeze(0)
        return activations
    
    else:
        print(f"Unknown activation type: {type(activations)}")
        return activations


class DetCAMModel(nn.Module):
    """Wrap the mmdet model class to facilitate handling of non-tensor
    situations during inference."""
    
    def __init__(self, cfg, checkpoint, score_thr, device='cuda:0'):
        super().__init__()
        self.device = device
        self.score_thr = score_thr
        self.detector = init_detector(cfg, checkpoint, device=device)
        self.cfg = self.detector.cfg
        self.input_data = None
        self.img = None
        self.return_loss = False
    
    def set_return_loss(self, return_loss):
        """Set whether to return loss for gradient-based methods."""
        self.return_loss = return_loss
    
    def set_input_data(self, img, bboxes=None, labels=None):
        """Set input data for the model."""
        self.img = img
        cfg = self.cfg.copy()
        
        # Build test pipeline
        test_pipeline = []
        for transform in cfg.test_dataloader.dataset.pipeline:
            if transform['type'] == 'LoadImageFromFile':
                continue
            test_pipeline.append(transform)
        
        h, w = self.img.shape[:2]
        data = {
            'img': self.img,
            'img_id': 0,
            'img_path': 'temp.jpg',  # dummy path
            'ori_shape': (h, w),
            'img_shape': (h, w),
            'scale_factor': np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            'scale': (w, h),
            'keep_ratio': True,
            'homography_matrix': np.eye(3, dtype=np.float32)
        }
        
        # Apply transforms
        pipeline = Compose(test_pipeline)
        data = pipeline(data)
        
        if self.return_loss and bboxes is not None and labels is not None:
            gt_instances = data['data_samples'].gt_instances
            gt_instances.bboxes = torch.from_numpy(bboxes).to(self.device)
            gt_instances.labels = torch.from_numpy(labels).to(self.device)
        
        if hasattr(data['inputs'], 'to'):
            data['inputs'] = data['inputs'].to(self.device)
            # Convert to float if it's a ByteTensor
            if data['inputs'].dtype == torch.uint8:
                data['inputs'] = data['inputs'].float()
        elif isinstance(data['inputs'], (list, tuple)):
            processed_inputs = []
            for inp in data['inputs']:
                if hasattr(inp, 'to'):
                    inp = inp.to(self.device)
                    if inp.dtype == torch.uint8:
                        inp = inp.float()
                processed_inputs.append(inp)
            data['inputs'] = processed_inputs
        
        if hasattr(data['data_samples'], 'to'):
            data['data_samples'] = data['data_samples'].to(self.device)
        elif isinstance(data['data_samples'], (list, tuple)):
            data['data_samples'] = [ds.to(self.device) if hasattr(ds, 'to') else ds 
                                   for ds in data['data_samples']]
        
        self.input_data = data
    
    def forward(self, *args, **kwargs):
        """Forward method required by PyTorch nn.Module"""
        return self.__call__(*args, **kwargs)
    
    def __call__(self, *args, **kwargs):
        """Main inference method."""
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            input_tensor = args[0]
            
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
            if input_tensor.dtype == torch.uint8:
                input_tensor = input_tensor.float()
            
            if self.return_loss:
                self.detector.eval() 
                input_tensor.requires_grad_(True)
            else:
                self.detector.eval()
            
            if self.return_loss:
                results = self.detector.predict(
                    input_tensor, 
                    [self.input_data['data_samples']]
                )
            else:
                with torch.no_grad():
                    results = self.detector.predict(
                        input_tensor, 
                        [self.input_data['data_samples']]
                    )
            
            pred_instances = results[0].pred_instances
            
            if len(pred_instances) == 0:
                return {"bboxes": np.array([]).reshape(0, 5), 'labels': np.array([]), 'segms': None}
            
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            # Combine bboxes and scores
            bboxes_with_scores = np.hstack([bboxes, scores.reshape(-1, 1)])
            
            segms = None
            if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
                segms = pred_instances.masks.cpu().numpy()
            
            if self.score_thr > 0:
                inds = scores > self.score_thr
                bboxes_with_scores = bboxes_with_scores[inds]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds]
            
            if len(bboxes_with_scores) == 0:
                bboxes_with_scores = np.array([]).reshape(0, 5)
                labels = np.array([])
                segms = None
            
            return {"bboxes": bboxes_with_scores, 'labels': labels, 'segms': segms}
        
        assert self.input_data is not None, "Input data not set. Call set_input_data() first."
        
        if self.return_loss:
            self.detector.train()
            inputs = self.input_data['inputs']
            if hasattr(inputs, 'requires_grad_'):
                inputs.requires_grad_(False)
            
            if isinstance(inputs, torch.Tensor) and inputs.dtype == torch.uint8:
                inputs = inputs.float()
            elif isinstance(inputs, (list, tuple)):
                processed_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor) and inp.dtype == torch.uint8:
                        inp = inp.float()
                    processed_inputs.append(inp)
                inputs = processed_inputs
            
            losses = self.detector.loss(
                inputs, 
                [self.input_data['data_samples']]
            )
            
            if isinstance(losses, dict):
                total_loss = sum(losses.values())
                return total_loss
            return losses
        else:
            self.detector.eval()
            
            inputs = self.input_data['inputs']
            
            if isinstance(inputs, torch.Tensor):
                if inputs.dim() == 3:  # C, H, W -> add batch dimension
                    inputs = inputs.unsqueeze(0)  # -> B, C, H, W
                if inputs.dtype == torch.uint8:
                    inputs = inputs.float()
            elif isinstance(inputs, (list, tuple)):
                processed_inputs = []
                for inp in inputs:
                    if isinstance(inp, torch.Tensor):
                        if inp.dim() == 3:
                            inp = inp.unsqueeze(0)
                        if inp.dtype == torch.uint8:
                            inp = inp.float()
                    processed_inputs.append(inp)
                inputs = processed_inputs
            
            with torch.no_grad():
                results = self.detector.predict(
                    inputs, 
                    [self.input_data['data_samples']]
                )
           
            pred_instances = results[0].pred_instances
            
            if len(pred_instances) == 0:
                return [{"bboxes": np.array([]).reshape(0, 5), 'labels': np.array([]), 'segms': None}]
            
            bboxes = pred_instances.bboxes.cpu().numpy()
            scores = pred_instances.scores.cpu().numpy()
            labels = pred_instances.labels.cpu().numpy()
            
            bboxes_with_scores = np.hstack([bboxes, scores.reshape(-1, 1)])
            segms = None
            if hasattr(pred_instances, 'masks') and pred_instances.masks is not None:
                segms = pred_instances.masks.cpu().numpy()
            
            if self.score_thr > 0:
                inds = scores > self.score_thr
                bboxes_with_scores = bboxes_with_scores[inds]
                labels = labels[inds]
                if segms is not None:
                    segms = segms[inds]
            
            if len(bboxes_with_scores) == 0:
                bboxes_with_scores = np.array([]).reshape(0, 5)
                labels = np.array([])
                segms = None
            
            return [{"bboxes": bboxes_with_scores, 'labels': labels, 'segms': segms}]


class DetAblationLayer(AblationLayer):

    def __init__(self):
        super(DetAblationLayer, self).__init__()
        self.activations = None

    def set_next_batch(self, input_batch_index, activations,
                       num_channels_to_ablate):
        """Extract the next batch member from activations, and repeat it
        num_channels_to_ablate times."""
        if isinstance(activations, torch.Tensor):
            return super(DetAblationLayer,
                         self).set_next_batch(input_batch_index, activations,
                                              num_channels_to_ablate)

        self.activations = []
        for activation in activations:
            activation = activation[
                input_batch_index, :, :, :].clone().unsqueeze(0)
            self.activations.append(
                activation.repeat(num_channels_to_ablate, 1, 1, 1))

    def __call__(self, x):
        """Go over the activation indices to be ablated, stored in
        self.indices.
        Map between every activation index to the tensor in the Ordered Dict
        from the FPN layer.
        """
        result = self.activations

        if isinstance(result, torch.Tensor):
            return super(DetAblationLayer, self).__call__(x)

        channel_cumsum = np.cumsum([r.shape[1] for r in result])
        num_channels_to_ablate = result[0].size(0)  # batch
        for i in range(num_channels_to_ablate):
            pyramid_layer = bisect.bisect_right(channel_cumsum,
                                                self.indices[i])
            if pyramid_layer > 0:
                index_in_pyramid_layer = self.indices[i] - channel_cumsum[
                    pyramid_layer - 1]
            else:
                index_in_pyramid_layer = self.indices[i]
            result[pyramid_layer][i, index_in_pyramid_layer, :, :] = -1000
        return result


class DetCAMVisualizer:
    """mmdet cam visualization class.
    Args:
        method_class: CAM method class.
        model (nn.Module): MMDet model.
        target_layers (list[torch.nn.Module]): The target layers
            you want to visualize.
        reshape_transform (Callable, optional): Function of Reshape
            and aggregate feature maps. Defaults to None.
        is_need_grad (bool): Whether to use gradient-based method.
        extra_params (dict): Extra parameters for CAM method.
    """

    def __init__(self,
                 method_class,
                 model,
                 target_layers,
                 reshape_transform=None,
                 is_need_grad=False,
                 extra_params=None):
        
        if extra_params is None:
            extra_params = {}
        
        # Use the fixed reshape_transform if None provided
        if reshape_transform is None:
            reshape_transform = reshape_transform
        
        if is_need_grad:
            # Gradient-based methods
            self.cam = method_class(
                model,
                target_layers,
                reshape_transform=reshape_transform,
            )
        else:
            # Gradient-free methods
            if method_class.__name__ == 'AblationCAM':
                self.cam = method_class(
                    model,
                    target_layers,
                    reshape_transform=reshape_transform,
                    batch_size=extra_params.get('batch_size', 1),
                    ablation_layer=extra_params.get('ablation_layer', None),
                    ratio_channels_to_ablate=extra_params.get('ratio_channels_to_ablate', 0.1)
                )
            else:
                self.cam = method_class(
                    model,
                    target_layers,
                    reshape_transform=reshape_transform,
                )
        
        self.classes = model.detector.dataset_meta.get('classes', None)
        if self.classes is None:
            # Fallback to CLASSES attribute if available
            self.classes = getattr(model.detector, 'CLASSES', [f'class_{i}' for i in range(80)])
        
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.is_need_grad = is_need_grad

    def switch_activations_and_grads(self, model):
        """Switch between activations and gradients for gradient-based methods."""
        if hasattr(self.cam, 'activations_and_grads'):
            self.cam.activations_and_grads.release()
        
        if self.is_need_grad:
            from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
            self.cam.activations_and_grads = ActivationsAndGradients(
                model, self.cam.target_layers, self.cam.reshape_transform)

    def __call__(self, img, targets, aug_smooth=False, eigen_smooth=False):
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
        return self.cam(img, targets, aug_smooth, eigen_smooth)[0, :]

    def show_cam(self,
                 image,
                 boxes,
                 labels,
                 grayscale_cam,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if with_norm_in_bboxes is True:
            boxes = boxes.astype(np.int32)
            renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
            images = []
            for x1, y1, x2, y2 in boxes:
                img = renormalized_cam * 0
                img[y1:y2,
                    x1:x2] = scale_cam_image(grayscale_cam[y1:y2,
                                                           x1:x2].copy())
                images.append(img)

            renormalized_cam = np.max(np.float32(images), axis=0)
            renormalized_cam = scale_cam_image(renormalized_cam)
        else:
            renormalized_cam = grayscale_cam

        cam_image_renormalized = show_cam_on_image(
            image / 255, renormalized_cam, use_rgb=False)

        image_with_bounding_boxes = self._draw_boxes(boxes, labels,
                                                     cam_image_renormalized)
        return image_with_bounding_boxes

    def _draw_boxes(self, boxes, labels, image):
        for i, box in enumerate(boxes):
            label = labels[i]
            color = self.COLORS[label]
            cv2.rectangle(image, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), color, 2)
            cv2.putText(
                image,
                self.classes[label], (int(box[0]), int(box[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                lineType=cv2.LINE_AA)
        return image


class DetBoxScoreTarget:
    """For every original detected bounding box specified in "bboxes",
    assign a score on how the current bounding boxes match it,
        1. In Bbox IoU
        2. In the classification score.
        3. In Mask IoU if ``segms`` exist.
    If there is not a large enough overlap, or the category changed,
    assign a score of 0.
    The total score is the sum of all the box scores.
    """

    def __init__(self,
                 bboxes,
                 labels,
                 segms=None,
                 match_iou_thr=0.5,
                 device='cuda:0'):
        assert len(bboxes) == len(labels)
        self.focal_bboxes = torch.from_numpy(bboxes).to(device=device)
        self.focal_labels = labels
        if segms is not None:
            assert len(bboxes) == len(segms)
            self.focal_segms = torch.from_numpy(segms).to(device=device)
        else:
            self.focal_segms = [None] * len(labels)
        self.match_iou_thr = match_iou_thr
        self.device = device

    def __call__(self, results):
        output = torch.tensor([0.0], device=self.device)
        
        # Handle empty results
        if len(results["bboxes"]) == 0 or results["bboxes"].size == 0:
            return output

        pred_bboxes = torch.from_numpy(results["bboxes"]).to(self.device)
        pred_labels = results["labels"]
        pred_segms = results["segms"]

        if pred_segms is not None and len(pred_segms) > 0:
            pred_segms = torch.from_numpy(pred_segms).to(self.device)

        for focal_box, focal_label, focal_segm in zip(self.focal_bboxes,
                                                      self.focal_labels,
                                                      self.focal_segms):
            ious = torchvision.ops.box_iou(focal_box[None],
                                           pred_bboxes[..., :4])
            index = ious.argmax()
            if ious[0, index] > self.match_iou_thr and pred_labels[
                    index] == focal_label:
                # TODO: Adaptive adjustment of weights based on algorithms
                score = ious[0, index] + pred_bboxes[..., 4][index]
                output = output + score

                if focal_segm is not None and pred_segms is not None:
                    segms_score = (focal_segm * pred_segms[index]).sum() / (
                        focal_segm.sum() + pred_segms[index].sum() + 1e-7)
                    output = output + segms_score
        return output


class FeatmapAM(EigenCAM):
    """Visualize Feature Maps.
    Visualize the (B,C,H,W) feature map averaged over the channel dimension.
    """

    def __init__(self,
                 model,
                 target_layers,
                 use_cuda=False,
                 reshape_transform=None):
        super(FeatmapAM, self).__init__(model, target_layers, use_cuda,
                                        reshape_transform)

    def get_cam_image(self, input_tensor, target_layer, target_category,
                      activations, grads, eigen_smooth):
        return np.mean(activations, axis=1)