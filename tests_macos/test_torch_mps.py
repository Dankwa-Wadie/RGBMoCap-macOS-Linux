import torch
from mocap import torch_mps

torch_mps.install()


def test_cuda_name_means_mps():
    assert torch.cuda.is_available()
    x = torch.zeros(3, device='cuda'); assert x.device.type == 'mps'
    y = torch.ones(2).cuda(); assert y.device.type == 'mps'
    z = torch.arange(4).to('cuda:0'); assert z.device.type == 'mps'
    w = torch.randn(2, 2).to(device=torch.device('cuda')); assert w.device.type == 'mps'
    m = torch.nn.Linear(4, 2).cuda(); assert next(m.parameters()).device.type == 'mps'
    m2 = torch.nn.Linear(4, 2).to('cuda'); assert next(m2.parameters()).device.type == 'mps'
    out = m(torch.randn(5, 4, device='cuda')); assert out.shape == (5, 2)
    assert x.new_zeros(2, device='cuda').device.type == 'mps'
    assert torch.zeros_like(x, device='cuda').device.type == 'mps'


def test_memory_api():
    free, total = torch.cuda.mem_get_info(); assert 0 <= free <= total and total > 0
    torch.cuda.reset_peak_memory_stats(); torch.randn(1024, 1024, device='cuda')
    assert torch.cuda.max_memory_allocated() > 0
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    assert torch.cuda.get_device_properties(0).total_memory == total
    e1 = torch.cuda.Event(enable_timing=True); e2 = torch.cuda.Event(enable_timing=True)
    e1.record(); torch.randn(10, device='cuda'); e2.record(); e2.synchronize()
    assert e1.elapsed_time(e2) >= 0


def test_autocast_and_load(tmp_path):
    with torch.autocast('cuda', dtype=torch.float16):
        r = torch.nn.Linear(4, 4).cuda()(torch.randn(2, 4, device='cuda'))
    assert r.dtype == torch.float32  # float32 by default, as validated on Windows
    with torch.cuda.amp.autocast():
        pass
    torch.save({'w': torch.ones(2)}, tmp_path / 'c.pt')
    assert torch.load(tmp_path / 'c.pt', map_location='cuda')['w'].device.type == 'mps'
    assert torch.load(tmp_path / 'c.pt')['w'].device.type == 'cpu'
    assert torch.OutOfMemoryError is torch.cuda.OutOfMemoryError


def test_seeding_is_reproducible():
    torch.manual_seed(0); torch.cuda.manual_seed(0); torch.cuda.manual_seed_all(0)
    a = torch.randn(4, device='cuda')
    torch.manual_seed(0)
    b = torch.randn(4, device='cuda')
    assert torch.equal(a.cpu(), b.cpu())


def test_float64_copy_to_cpu():
    x = torch.ones(2, 2, device='cuda')
    y = x.to('cpu', torch.float64)
    assert y.device.type == 'cpu' and y.dtype == torch.float64
    z = x.to(device='cpu', dtype=torch.float64)
    assert z.device.type == 'cpu' and z.dtype == torch.float64
