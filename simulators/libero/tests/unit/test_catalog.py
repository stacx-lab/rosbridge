from collections import Counter

from rosbridge.catalog import resolve, tasks


def test_official_catalog_has_unique_ids_and_contiguous_suite_indices():
    entries = list(tasks(name='libero').values())
    sizes = {'libero_spatial': 10, 'libero_object': 10, 'libero_goal': 10,
             'libero_10': 10, 'libero_90': 90}
    assert Counter(t['suite'] for t in entries) == sizes
    for suite, size in sizes.items():
        selected = [t for t in entries if t['suite'] == suite]
        assert [t['task_index'] for t in selected] == list(range(size))
        assert len({t['task_name'] for t in selected}) == size
    assert resolve('libero')[2]['id'] == 'libero/libero-spatial-01'
    for task in entries:
        assert resolve(task['id'])[2] == task
