import sys

def GetHumanReadable(size, precision=2):
    """Takes a byte sized input and computes the closest
    human readable format, e.g., in megabytes etc."""
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    suffixIndex = 0
    while size > 1024 and suffixIndex < 4:
        suffixIndex += 1 
        size = size / 1024
    return "%.*f%s" % (precision, size, suffixes[suffixIndex])


def compute_network_pass_size(model):
    """Computes the size of a network pass in bytes using cached
    parameters as well as gradients"""
    num_bytes = 0

    print('Adding layer caches for forward pass:')
    for layer in model.cache.keys():
        # Add size of forward caches
        key_num_bytes = 0
        for value in model.cache[layer]:
            value_num_bytes = sys.getsizeof(value)
            key_num_bytes += value_num_bytes
        num_bytes += key_num_bytes

        print(layer, key_num_bytes)

    print('\nAdding layer gradients for backward pass:')
    for key in model.grads.keys():
        # Add size of backward gradients
        key_num_bytes = sys.getsizeof(model.grads[key])
        num_bytes += key_num_bytes
        print(key, key_num_bytes)
       
    return num_bytes
