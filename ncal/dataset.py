"""N-Caltech101 classification adapter for SpikingJelly's legacy resource list."""
from spikingjelly.datasets.n_caltech101 import NCaltech101 as _NCaltech101


class NCaltech101(_NCaltech101):
    @staticmethod
    def resource_url_md5():
        # Classification only extracts Caltech101.zip. Retain its upstream MD5,
        # without requiring annotations or obsolete README download resources.
        resources = [resource for resource in _NCaltech101.resource_url_md5()
                     if resource[0] == 'Caltech101.zip']
        if len(resources) != 1:
            raise RuntimeError('Expected one Caltech101.zip resource from SpikingJelly')
        return resources
