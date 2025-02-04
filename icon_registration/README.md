Directory:

networks.py

the network architectures used in the ICON paper. They take in tensors (images) and output tensors (various interpretations)

network_wrappers.py:

these classes are pytorch 'functionals': they take in pytorch tensors (images) and output first class python functions that map tensors of coordinates to tensors of coordinates.
