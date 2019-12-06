#Network-Construction

Functions for creating complex neural network structures with ease. 

__linear\_struct(in\_shape,conv\_layers,FC\_layers,\*args,\*\*kwargs)__: This is the base constructor of all other networks. All networks are either wholly or partially comprised of linear blocks of convolution layers. It is mainly designed to be able to produce models for image classifiers, but need not necessarily do so as it can also be used for regression models and MLPs.

__branched\_struct(in\_size, branches,\*args.\*\*kwargs):__ This will join together several models in parallel where each will have the same input and each model's output will be concatenated to produce a combined output. 

*im_size* is a tuple describing the dimensions of the input tensor for each instance in the form (w,h,c),[w,h,c],(w,h) or [w,h] wherre w is the width, h the height and c the number of channels. *branches* is a list of models that are to be joined together in this structure. 

Prudent implementation of this function will involve branches that are compatible with the stated input tensor and that have suitable dimensions for concatenation. Nonetheless, this is not always easy to achieve, especially when different pooling, padding and convolution operations are applied in each branch. Using the "fix_mismatches" argument will compensate for this. For the inputs, it will implement a resizing, padding or crop operation to match the size of the stated input with the expected input of each branch. For the output, a suitable tuple for the height and width must be supplied so that concatenation can occur. This is done using the "out_shape" keyword argument with the value being that two element list or tuple. If the "summary" argument is supplied, then it will print a summary to screen of the branched structure. This will not include the detailes of each model, but only the components used in constructing the parellel structure of this branch. There are automatic rules for which of the three resizing operations will be applied, but if one wants to specify a given rule for all branched, one can use the "fit_rules" string argument with a value from the options "resize","zero_pad", "copy_pad" or "crop". If one wants to specify the same rules for input and output for each branch, the value is a list of the previous options with the same length as number of branches. If one wants to specify different rules for input and output adjustments, the value should be a list of two-element lists or tuples with entries from these options.

** block_struct **

** network_struct **
