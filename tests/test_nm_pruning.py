import torch

from nanopointllm.compression.nm_pruning import nm_mask, validate_nm_mask


def test_nm_mask_prunes_exactly_two_of_four_per_row_group():
    weight = torch.tensor([
        [1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0],
        [8.0, 7.0, 6.0, 5.0, 1.0, 2.0, 3.0, 4.0],
    ])
    mask = nm_mask(weight, prune_n=2, prune_m=4)
    assert validate_nm_mask(mask, prune_n=2, prune_m=4)
    assert mask[0].tolist() == [True, False, True, False, False, True, False, True]


def test_wanda_metric_can_change_magnitude_selection():
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    magnitude = nm_mask(weight, prune_n=2, prune_m=4)
    wanda = nm_mask(
        weight,
        prune_n=2,
        prune_m=4,
        input_second_moment=torch.tensor([100.0, 100.0, 0.01, 0.01]),
    )
    assert magnitude.tolist() == [[True, True, False, False]]
    assert wanda.tolist() == [[False, False, True, True]]
