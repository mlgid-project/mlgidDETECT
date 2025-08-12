import torch
from torch import Tensor
from torchvision.ops import box_iou, nms
from typing import Tuple
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def _format_str(v) -> str:
    if isinstance(v, float):
        if v >= 0.01:
            return f'{v:.2f}'
        return f'{v:.2e}'
    return str(v)

def filter_nms(predictions: Tensor, scores: Tensor, level: float = 0.1):
    if not scores.numel():
        return predictions, scores

    indices = nms(predictions, scores, level)
    return predictions[indices], scores[indices]

def standard_filter(predictions: Tensor, scores: Tensor,
                    nms_level: float = 0.1,
                    score_level: float = 0.8,
                    ):
    predictions, scores = filter_score(predictions, scores, score_level)
    predictions, scores = filter_nms(predictions, scores, nms_level)

    return predictions, scores


def get_dq_mtx(boxes1, boxes2=None):
    if boxes2 is None:
        boxes2 = boxes1
    dq = (boxes1[:, 0] + boxes1[:, 2])[None] / 2 - (boxes2[:, 0] + boxes2[:, 2])[:, None] / 2
    return dq

def filter_score(predictions: Tensor, scores: Tensor, level: float = 0.1):
    if not level:
        return predictions, scores

    if not scores.numel():
        return predictions, scores

    indices = scores > level
    return predictions[indices], scores[indices]

def merge_boxes(
        boxes, scores,
        min_iou: float = 0.3,
        max_q: float = 4.,
        min_score: float = 0.2,
        mode: str = 'mean-quantile',
        quantile: float = 0.8,
):
    boxes, scores = filter_score(boxes, scores, min_score)

    iou_mtx = box_iou(boxes, boxes) > min_iou
    q_mtx = get_dq_mtx(boxes).abs() < max_q

    graph = csr_matrix((iou_mtx * q_mtx).detach().cpu().numpy())
    num_clusters, labels = connected_components(graph, directed=False, return_labels=True)

    merged_scores = torch.stack([
        scores[labels == label].max() for label in range(num_clusters)
    ], dim=0)

    if mode == 'mean-quantile':
        score_weights = [
            scores[labels == label] / scores[labels == label].sum()
            for label in range(num_clusters)
        ]

        qs, a_qs = (boxes[:, :2] + boxes[:, 2:]).T / 2
        ws, a_ws = (boxes[:, 2:] - boxes[:, :2]).T / 2

        qs, ws, a_qs, a_ws = torch.stack([
            torch.stack([
                (qs[labels == label] * weight).sum(0),
                (ws[labels == label] * weight).sum(0),
                (a_qs[labels == label] * weight).sum(0),
                torch.quantile(a_ws[labels == label], quantile),
                #                 a_ws[labels == label].max(0).values,
            ]) for label, weight in enumerate(score_weights)
        ], 1)

        merged_boxes = torch.stack([
            qs - ws, a_qs - a_ws, qs + ws, a_qs + a_ws
        ], dim=-1)

    elif mode == 'mean':
        merged_boxes = torch.stack([
            (
                    boxes[labels == label] * scores[labels == label][:, None]
            ).sum(0) / scores[labels == label].sum()
            for label in range(num_clusters)
        ], dim=0)

    elif mode == 'max':
        merged_boxes = torch.stack([
            torch.cat([
                boxes[labels == label][:, :2].min(0).values,
                boxes[labels == label][:, 2:].max(0).values,
            ], 0) for label in range(num_clusters)
        ])
    else:
        raise ValueError(f'Unknown mode {mode}, should be one of `mean`, `max`, `mean-quantile`.')

    return merged_boxes, merged_scores



class Postprocessing(object):
    def __call__(self, predictions: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def __add__(self, other: "Postprocessing"):
        if not isinstance(other, Postprocessing):
            raise NotImplemented

        return PostprocessingPipeline(self, other)

    def __repr__(self):
        args = ', '.join(f'{k}={_format_str(v)}' for k, v in vars(self).items())
        return f"{self.__class__.__name__}({args})"
    
class PostprocessingPipeline(Postprocessing):
    def __init__(self, *postprocessing):
        self._postprocessing = postprocessing

    def __call__(self, predictions: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        for p in self._postprocessing:
            predictions, scores = p(predictions, scores)
        return predictions, scores

    def __add__(self, other: "Postprocessing"):
        """ if not isinstance(other, Postprocessing):
            raise NotImplemented """

        return PostprocessingPipeline(*self._postprocessing, other)


class SmallQFilter(Postprocessing):
    def __init__(self, min_q_pix: float = 1.):
        self.min_q_pix = min_q_pix

    def __call__(self, predictions: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        qs = (predictions[..., 0] + predictions[..., 2]) / 2
        indices = qs >= self.min_q_pix
        return predictions[indices], scores[indices]
    
class MergeBoxesPostprocessing(Postprocessing):
    def __init__(self,
                 min_iou: float = 0.3,
                 max_q: float = 4.,
                 min_score: float = 0.5,
                 mode: str = 'mean-quantile',
                 quantile: float = 0.8,
                 ):
        self.min_iou = min_iou
        self.max_q = max_q
        self.min_score = min_score
        self.mode = mode
        self.quantile = quantile

    def __call__(self, predictions: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        return merge_boxes(
            predictions, scores,
            max_q=self.max_q,
            min_score=self.min_score,
            mode=self.mode,
            quantile=self.quantile,
        )
    
class StandardPostprocessing():
    def __init__(self,
                 nms_level: float = 0.1,
                 score_level: float = 0.8,
                 ):
        self.nms_level = nms_level
        self.score_level = score_level

    def __call__(self, predictions: Tensor, scores: Tensor) -> Tuple[Tensor, Tensor]:
        return standard_filter(
            predictions, scores, nms_level=self.nms_level, score_level=self.score_level
        )
