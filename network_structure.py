# Functions for the creation of complex neural networks

from model_analysis import *

def linear_struct(in_shape,conv_layers,FC_layers,*args,**kwargs):
    from keras.layers import Input
    # First select the correct form of the input be either obtaining the correct vector or using the im_type keyword.
    if "im_type" in kwargs:
        if kwargs["im_type"] in ["color","colour"]:
            in_shape=(in_shape[0],in_shape[1],3,)
        elif kwargs["im_type"] in ["grayscale","greyscale"]:
            in_shape=(in_shape[0],in_shape[1],1,)
        else:
            print("Image Type Not Recognised!")
    else:
        if len(in_shape)==2:
            in_shape=(in_shape[0],in_shape[1],1,)
        elif len(in_shape)==3:
            in_shape=(in_shape[0],in_shape[1],in_shape[2],)
        else:
            print("This input is a nonsense shape!")
    # We get the unique identifier for this substructure if applicable
    if "annotation" in kwargs:
        annotation=kwargs["annotation"]
    else:
        annotation=""
    # We use kwargs to set how we are applying padding, initialisation of weights and activation function. If not given,
    # the defaults are random glorot distribution and ReLu.
    if "kernel_init" in kwargs:
        kernel_init_rule=kwargs["kernel_init"]
    else:
        kernel_init_rule="glorot_uniform"
    if "bias_init" in kwargs:
        bias_init_rule=kwargs["bias_init"]
    else:
        bias_init_rule="zeros"
    if "activation" in kwargs:
        act_rule=kwargs["activation"]
    else:
        act_rule="relu"

    # We need a dictionary to refer to the layers by name in a general way - its length will vary

    layers_dict={}
    layers_dict["main_input"]=Input(shape=in_shape,name="Input"+"_"+annotation)
    connecting="main_input" # This will be a running record of what we are feeding into the next layer

    # We can apply batch normalisation and/or dropout to layers using the "BN" and "DropOut" kwargs. This specifies which
    # layers (excluding pooling layers) where both or either technique is applied.

    # CHECK IF THIS WORKS

    if conv_layers==None:
        conv_layers=[]
    if FC_layers==None:
        FC_layers=[]

    if "BN" in args:
        from keras.layers import BatchNormalization as BN
        kwargs["BN"]={i:True for i in range(len(conv_layers)+len(FC_layers))}
    elif "BN" in kwargs:
        BN_rules=kwargs["BN"]
        from keras.layers import BatchNormalization as BN
        if kwargs["BN"]=="all":
            kwargs["BN"]={i:True for i in range(len(conv_layers)+len(FC_layers))}
        else:
            # This is not tested yet
            kwargs["BN"]={i:True for i in range(len(conv_layers)+len(FC_layers)) if i in kwargs["BN"]}+{i:None for i in range(len(conv_layers+FC_layers)) if i not in kwargs["BN"]}
    else:
        kwargs["BN"]={i:None for i in range(len(conv_layers)+len(FC_layers))}
    if "DropOut" in kwargs:
        DO_rules=kwargs["DropOut"]
        from keras.layers import Dropout as DO

    # After this initial setup we can start creating the layers along with their pooling/BN/DropOut options.

    # Apply convolution layers first
    if conv_layers==[]:
        pass
    else:
        if "octave" in args:
            from keras_octave_conv import OctaveConv2D as cn
            if "oct_alpha" in kwargs:
                rat_rule=kwargs["oct_alpha"]
                if rat_rule not in [0.125,0.25,0.5,0.75,1]:
                    rat_rule=0.5
            else:
                rat_rule=0.5
            if "octave_scale" not in kwargs:
                oct_scale=2
            else:
                oct_scale=kwargs["octave_scale"]
        else:
            from keras.layers import Conv2D as cn
        # The following kwargs are only relevant where we have convolutions
        if "padding" in kwargs:
            pad_rule=kwargs["padding"]
        else:
            pad_rule=None

        # This strides kwarg will be a dictionary for each stride type that will give the layer of the convolution.
        # Note that mostly we only have a different stride for the initial convolution layer.

        if "strides" in kwargs:
            strides_by_layer={i:(kwargs["strides"][i],kwargs["strides"][i]) for i in range(len(conv_layers))}
        else:
            strides_by_layer={i:(1,1) for i in range(len(conv_layers))}

         # We take pooling rules for each layer from the kwargs or apply a default pooling factor of 2.

        if "pooling" in kwargs:
            if kwargs["pooling"]==None:
                pool_rule={i:None for i in range(len(conv_layers))}
            else:
                pool_rule={i:(p,p) for p in kwargs["pooling"] for i in range(len(kwargs["pooling"]))}
        else:
                pool_rule={i:(2,2) for i in range(len(conv_layers))}
        # We take the type of pooling if specified - default is MaxPooling2D - note that we apply this globally to all layers
        if "pooling_type" in kwargs:
            if kwargs["pooling_type"]=="average" or kwargs["pooling_type"]==None:
                from keras.layers import AveragePooling2D as pool
            elif kwargs["pooling_type"]=="max":
                from keras.layers import MaxPooling2D as pool
            elif kwargs["pooling_type"]=="global":
                from keras.layers import GlobalMaxPooling2D as pool
        else:
            from keras.layers import MaxPooling2D as pool

        for c in range(len(conv_layers)):

            # Set size of convolutions in current layer

            if len(conv_layers[c])==1:
                conv_size=(3,3)
            else:
                conv_size=tuple(2*[conv_layers[c][1]])

            # If using octave convolutions we need to split each layer into hi and lo components

            if "octave" in args:
                # for the first convolution, we need to create the split. For pooling, BN and DropOut, they need to be
                # applied seperately for the hi and lo portions
                if c==0:
                    layers_dict["hi_1"], layers_dict["lo_1"] = cn(filters=conv_layers[0][0],
                                               kernel_size=conv_size[0],octave=oct_scale, ratio_out=rat_rule)(layers_dict[connecting])
                    hi_connect,lo_connect= "hi_1","lo_1"
                    if pool_rule[0] != None:
                        layers_dict["hi_1 (pooled)"] = pool(pool_rule[c],name="hi_pool_layer" + str(c + 1) + annotation)(
                            layers_dict[hi_connect])
                        layers_dict["lo_1 (pooled)"] = pool(pool_rule[c],name="lo_pool_layer" + str(c + 1) + annotation)(
                            layers_dict[lo_connect])
                        hi_connect="hi_1 (pooled)"
                        lo_connect="lo_1 (pooled)"

                    if "BN" in kwargs:
                        if kwargs["BN"][0] != None:
                            layers_dict["BN_hi_1"] = BN(name="BN_hi_1" + annotation)(
                                layers_dict[hi_connect])
                            hi_connect= "BN_hi_1"
                            layers_dict["BN_lo_1"] = BN(name="BN_lo_1" + annotation)(
                                layers_dict[lo_connect])
                            lo_connect= "BN_lo_1"

                    if "DropOut" in kwargs:
                        if kwargs["DropOut"][0] != None:
                            layers_dict["DropOut_hi_1"] = DO(DO_rules[0],
                                                                      name="DropOut_hi_1" + annotation)(
                                layers_dict[hi_connect])
                            connecting = "DropOut_hi_1"
                            layers_dict["DropOut_lo_1"] = DO(DO_rules[0],
                                                                      name="DropOut_lo_1" + annotation)(
                                layers_dict[lo_connect])
                            connecting = "DropOut_lo_1"

                # for the last convolution, we need to join the hi and lo together again.
                elif c==(len(conv_layers)-1):
                    layers_dict["Final_Octave_layer"] = cn(filters=conv_layers[c][0], kernel_size=conv_size[0],
                                                                   ratio_out=0.0)([layers_dict[hi_connect],
                                                                                   layers_dict[lo_connect]])
                    connecting = "Final_Octave_layer"
                    if pool_rule[c] != None:
                        layers_dict["Pooling_layer_" + str(c + 1)] = pool(pool_rule[c],
                                                                          name="Pool_layer" + str(c + 1) + annotation)(
                            layers_dict[connecting])
                    connecting = "Pooling_layer_" + str(c + 1)

                    if "BN" in kwargs:
                        if kwargs["BN"][c] != None:
                            layers_dict["BN_Final_Oct"] = BN(name="BN_Final_Octave" + annotation)(
                                layers_dict[connecting])
                            connecting = "BN_Final_Oct"

                    # Otherwise we need to change the splits between convolutions
                else:
                    layers_dict["hi_"+str(c+1)], layers_dict["lo_"+str(c+1)] = cn(filters=8, kernel_size=3)([layers_dict[hi_connect],
                                                                                                                       layers_dict[lo_connect]])
                    hi_connect,lo_connect= "hi_"+str(c+1),"lo_"+str(c+1)

                    if pool_rule[c] != None:
                        layers_dict["hi_"+str(c+1)+"(pooled)"] = pool(pool_rule[c],name="hi_pool_layer" + str(c + 1) + annotation)(
                            layers_dict[hi_connect])
                        layers_dict["lo_"+str(c+1)+"(pooled)"] = pool(pool_rule[c],name="lo_pool_layer" + str(c + 1) + annotation)(
                            layers_dict[lo_connect])
                        hi_connect="hi_"+str(c+1)+"(pooled)"
                        lo_connect="lo_"+str(c+1)+"(pooled)"

                    if "BN" in kwargs:
                        if kwargs["BN"][c] != None:
                            layers_dict["BN_hi_"+str(c+1)] = BN(name="BN_hi_"+str(c+1) + annotation)(layers_dict[hi_connect])
                            hi_connect= "BN_hi_"+str(c+1)
                            layers_dict["BN_lo_"+str(c+1)] = BN(name="BN_lo_"+str(c+1) + annotation)(layers_dict[lo_connect])
                            lo_connect= "BN_lo_"+str(c+1)

                    if "DropOut" in kwargs:
                        if kwargs["DropOut"][c] != None:
                            layers_dict["DropOut_hi_"+str(c+1)] = DO(DO_rules[c],
                                                                      name="DropOut_hi_"+str(c+1) + annotation)(layers_dict[hi_connect])
                            connecting = "DropOut_hi_"+str(c+1)
                            layers_dict["DropOut_lo_"+str(c+1)] = DO(DO_rules[c],
                                                                      name="DropOut_lo_"+str(c+1) + annotation)(
                                layers_dict[lo_connect])
                            connecting = "DropOut_lo_"+str(c+1)

               # ADD DROPOUT TO EACH OCTAVE TYPE
            else:
                # If not an in implementation of Octave Convolutions, we use normal convolutions
                if pad_rule!=None:
                    layers_dict["Conv_layer_"+str(c+1)]=cn(conv_layers[c][0], conv_size, activation=act_rule, padding=pad_rule,
                                                           kernel_initializer=kernel_init_rule, bias_initializer=bias_init_rule,
                                                           strides=strides_by_layer[c], name="Conv_layer_"+str(c+1)+annotation)(layers_dict[connecting])
                    connecting="Conv_layer_"+str(c+1)
                else:
                    layers_dict["Conv_layer_"+str(c+1)]=cn(conv_layers[c][0], conv_size, activation=act_rule,
                                                           kernel_initializer=kernel_init_rule, bias_initializer=bias_init_rule,
                                                           strides=strides_by_layer[c], name="Conv_layer_"+str(c+1)+annotation)(layers_dict[connecting])
                    connecting="Conv_layer_"+str(c+1)
                if pool_rule[c]!=None:
                    layers_dict["Pooling_layer_"+str(c+1)]=pool(pool_rule[c], name="Pool_layer"+str(c+1)+annotation)(layers_dict[connecting])
                    connecting="Pooling_layer_"+str(c+1)

                if "BN" in kwargs:
                    if kwargs["BN"][c]!=None:
                        layers_dict["BN_"+str(c+1)]=BN(name="BN_"+str(c+1)+annotation)(layers_dict[connecting])
                        connecting="BN_"+str(c+1)

                # NEED TO ADD DROPOUT LEVEL

                if "DropOut" in kwargs:
                    if kwargs["DropOut"][c]!=None:
                        layers_dict["DropOut_"+str(c+1)]=DO(DO_rules[c],
                                                            name="DropOut_"+str(c+1)+annotation)(layers_dict[connecting])
                        connecting="DropOut_"+str(c+1)

    # Now we add the dense layers

    if FC_layers==[]:
        pass
    else:
        from keras.layers import Dense as FC, Flatten
        layers_dict["Flattened"]=Flatten(name="Flattened"+annotation)(layers_dict[connecting])
        connecting="Flattened"
        layer_offset=len(conv_layers) # We count from the last convolutional layer
        for f in range(len(FC_layers)):
            layers_dict["Dense_"+str(f+1+layer_offset)]=FC(FC_layers[f],name="Dense_"+str(f+1+layer_offset)+annotation,
                                                          kernel_initializer=kernel_init_rule, bias_initializer=bias_init_rule,
                                                          activation=act_rule)(layers_dict[connecting])
            connecting="Dense_"+str(f+1+layer_offset)

            if "BN" in kwargs:
                if kwargs["BN"][layer_offset+f]!=None:
                    layers_dict["BN_"+str(layer_offset+f+1)]=BN(name="BN_"+str(layer_offset+f+1)+annotation)(layers_dict[connecting])
                    connecting="BN_"+str(layer_offset+f+1)

            # NEED TO ADD DROPOUT LEVEL

            if "DropOut" in kwargs:
                if kwargs["DropOut"][layer_offset+f]!=None:
                    layers_dict["DropOut_"+str(layer_offset+f+1)]=DO(DO_rules[layer_offset+f],
                                                                     name="DropOut_"+str(layer_offset+f+1)+annotation)(layers_dict[connecting])
                    connecting="DropOut_"+str(layer_offset+f+1)

    # Attach classifier layer if required

    if "classes" in kwargs:
        from keras.layers import Dense as FC
        if kwargs["classes"].__class__==int:
            if kwargs["classes"]==2:
                layers_dict["Binary_classifier"]=FC(2,name="Binary_classifier"+annotation,
                                                          kernel_initializer=kernel_init_rule, bias_initializer=bias_init_rule,
                                                          activation="sigmoid")(layers_dict[connecting])
                connecting="Binary_classifier"
                if "DropOut" in kwargs:
                    if kwargs["DropOut"][-1]!=None:
                        layers_dict["DropOut_classifier"]=DO(DO_rules[-1],name="DropOut_classifier"+annotation)(layers_dict[connecting])
                        connecting="DropOut_classifier"

            elif kwargs["classes"]>2:
                layers_dict["Softmax_classifier"]=FC(kwargs["classes"],name="Softmax_classifier"+annotation,
                                                          kernel_initializer=kernel_init_rule, bias_initializer=bias_init_rule,
                                                          activation="sigmoid")(layers_dict[connecting])
                connecting="Softmax_classifier"
                if "DropOut" in kwargs:
                    if kwargs["DropOut"][-1]!=None:
                        layers_dict["DropOut_classifier"]=DO(DO_rules[-1],name="DropOut_classifier"+annotation)(layers_dict[connecting])
                        connecting="DropOut_classifier"
            else:
                pass

    if "feature_extract" in kwargs and FC_layers==[]:
        extract_rule=kwargs["feature_extract"]
        if extract_rule=="tensor":
            pass
        else:
            if extract_rule=="average":
                from keras.layers import GlobalAveragePooling2D as extract
            else:
                from keras.layers import GlobalMaxPooling2D as extract
            layers_dict["Extract"]=extract(name="Extract"+annotation)(layers_dict[connecting])
            connecting="Extract"

    # Now we create the model by specifying input and output. The connecting label gives us the last layer applied.

    from keras.models import Model

    full_model=Model(name="Linear"+annotation,inputs=layers_dict["main_input"],outputs=layers_dict[connecting])

    if "summary" in args:
        print(full_model.summary())

    return full_model

# RANDOM MODEL GENERATOR

def random_struct(inputs,input_types,stride_options,convolution_lists,pooling_options,pooling_types,
                  fc_options,batch_no_or_yes, dropout_options,classes,feature_extracts,*args,**kwargs):
    input_choice=[choice(inputs),choice(input_types)]
    print("Random_linear_"+kwargs["annotation"])
    print()
    print("Inputs:",input_choice[0],input_choice[1])
    conv_choice=choice(convolution_lists)
    fc_choice=choice(fc_options)
    while conv_choice==[] and fc_choice==[]:
        conv_choice=choice(convolution_lists)
        fc_choice=choice(fc_options)
    stride_choice=[choice(stride_options)]+[1]*(len(conv_choice)-1)
    if stride_choice[0]!=1 and conv_choice!=None:
        print("Initial Stride:",stride_choice[0])
    pooling_choices=[choice(pooling_options),choice(pooling_types)]
    for c in range(len(conv_choice)):
        print("Convolution layer ",str(c+1),"contains",conv_choice[c][0],"convolutions of size",conv_choice[c][1])
        if pooling_choices[0]!=None:
            if pooling_choices[0][c]==None:
                print("Pooling not Applied")
            else:
                print("Pooling factor of",pooling_choices[0][c], "using",pooling_choices[1],"pooling.")
        else:
            print("Pooling not Applied")
    print("Numbers of Nodes in Dense Layers:",fc_choice)
    bn_choice=choice(batch_no_or_yes)
    if bn_choice=="BN":
        print("Batch Normalisation Applied")
    else:
        print("Batch Normalisation Not Applied")
    do_choice=[choice(dropout_options) for i in range(len(conv_choice)+len(fc_choice))]
    class_no=choice(classes)
    if class_no!=None:
        print("Classes:",class_no)
        classifier_DO=choice(dropout_options)
        do_choice.append(classifier_DO)
        extract_choice=None
    else:
        print("Feature Extractor Only")
        extract_choice=choice(feature_extracts)
        print("Extract Features using",extract_choice)
    print("Dropout per layer:",do_choice)
    print()
    return linear_struct([input_choice[0],input_choice[0]],conv_choice,fc_choice,bn_choice,im_type=input_choice[1],
                         pooling=pooling_choices[0], strides=stride_choice,pooling_type=pooling_choices[1],DropOut=do_choice,
                         classes=class_no,feature_extract=extract_choice,*args,**kwargs)


# EXAMPLE 2

#Chooses from 1.2-1.5M possible options

'''from random import choice

inputs=[150,250,350]
input_types=["color","grayscale"]
stride_options=[1,1,1,2]    # 1 in 4 chance of initial stride of 2
convolution_lists=[[],[[20,3],[20,3],[20,3]],[[20,3],[20,5],[20,7]],[[20,7],[30,5],[40,3]],[[30,3],[50,3],[30,3]],
                   [[40,5],[30,3],[40,3]],[[100,3],[80,3],[60,3]],[[20,11],[40,7],[30,3]]]
pooling_options=[[None,2,2],[None,3,2],None,[2,2,2],[3,2,2],[3,None,3]]
pooling_types=["average","max"]
fc_options=[[],[20],[40],[60],[20,20],[40,40],[60,60],[60,20],[40,60]]
batch_no_or_yes=["BN","No_BN"]
dropout_options=[None,0.2,0.3,0.5]
classes=[None,2,4,8,16,32]
feature_extracts=["tensor","max","average"]
model_dict={}

for i in range(50):
    model_dict["model_"+str(i+1)]=random_struct(inputs,input_types,stride_options,convolution_lists,pooling_options,pooling_types,
                  fc_options,batch_no_or_yes, dropout_options,classes,feature_extracts,annotation="_test_net_"+str(i+1))
for m in model_dict:
    model_dict[m].name=m
    model_info(model_dict[m],"diagram","display_tensors","display_names","save_summaries",save_dir="C:\\Users\\the_n\\OneDrive\\Python Programs\\Working Branch\\Random Models")'''

# BRANCH STRUCTURE
from keras.layers import concatenate, Input, Lambda
from keras.models import Model
from tensorflow import image, convert_to_tensor

# RESIZE LAYER

def branched_struct(in_size,branches,*args,**kwargs):
    if "annotate_branches" in kwargs:
        annotation=kwargs["annotate_branches"]
    else:
        annotation=""

    if "fix_mismatches" in args:
        r_count=1
    # Create input for all branches
    if len(in_size)==2 and (in_size.__class__==tuple or in_size.__class__==tuple):
        common_input = Input(shape=(in_size[0], in_size[1], 1,), name="Input" + "-" + annotation)
    elif len(in_size)==3 and (in_size.__class__==tuple or in_size.__class__==tuple):
        common_input = Input(shape=(in_size[0], in_size[1], in_size[2],), name="Input" + "-" + annotation)
    else:
        print("Input not of valid shape!")

    # create dictionary of branches to add to. This tracks the connecting layer for all branches and
    # is initially set to the common_input
    branch_dict = {"branch_" + str(b + 1) + annotation:common_input for b in range(len(branches))}

    # Resize branch input to match branch inputs if necessary:
    for b in range(len(branches)):
        if list(branches[b].input.shape)!= [None]+list(in_size):
            if "fix_mismatches" not in args:
                print("ERROR: input_mismatch in", branches[b].name)
                return None
            else:
                branch_dict["branch_" + str(b + 1) + annotation]=Lambda(lambda x: image.resize(x,branches[b].input.shape[1:-1],method=image.ResizeMethod.BICUBIC,preserve_aspect_ratio=False),name="Resize_"+str(r_count)+annotation)(common_input)
                branch_dict["branch_" + str(b + 1) + annotation]=branches[b](branch_dict["branch_" + str(b + 1) + annotation])
                r_count+=1
        else:
            branch_dict["branch_" + str(b + 1) + annotation]=branches[b](branch_dict["branch_" + str(b + 1) + annotation])

    # Check if there is a common output
    out_shapes=[list(branch_dict["branch_" + str(i) + annotation].shape[:-1]) for i in range(1, len(branch_dict) + 1)]
    out_test=False not in [out_shapes[i]==out_shapes[0] for i in range(1,len(out_shapes))]
    # If the test dor a common output fails, we need to resize the outputs. This can be done by resizing or padding
    if out_test==False:
        if "fix_mismatches" not in args:
            print("ERROR: Branch outputs not matching.")
            return None
        else:
            if "out_shape" not in kwargs:
                print("ERROR: out_shape keyword must be given")
                return None
            else:
                out_shape=convert_to_tensor(kwargs["out_shape"])
                # START HERE!
                for b in branch_dict:
                    if list(branch_dict[b].shape)[1:-1] != list(out_shape):    # If the output shape is not as expected, resize the output to fit (later add padding options)
                        branch_dict[b] = Lambda(lambda x: image.resize(branch_dict[b],out_shape, method=image.ResizeMethod.BICUBIC, preserve_aspect_ratio=False),name="Resize_"+str(r_count)+annotation)(branch_dict[b])
                        r_count+=1
                    else:
                        pass
                output = concatenate([branch_dict[b] for b in branch_dict], axis=-1)
    else:
        output=concatenate([branch_dict[b] for b in branch_dict],axis=-1)

    full_model = Model(inputs=common_input, outputs=[output],name="Branch_model" + annotation,)
    return full_model


branch1=linear_struct((50,50),[[20,3]],None,annotation="_branch_1")
branch2=linear_struct((100,100),[[10,5],[20,3]],None,annotation= "_branch_2")
branch3=linear_struct((100,100),[[5,3],[1,3]],None,annotation="_branch_3")
branch4=linear_struct((150,150),[[20,3],[20,3],[20,3]],None,annotation="_branch_4")
branch5=linear_struct((75,75),[[20,3]],None,annotation="_branch_5")
branches=[branch1,branch2,branch3,branch4,branch5]
for b in branches:
    #print(b.summary())
    pass

test_model=branched_struct((100,100,1),branches,"fix_mismatches",out_shape=(23,23))
model_info(test_model,"diagram","display_tensors","display_names","save_summaries")
for b in branches:
    model_info(b, "diagram", "display_tensors", "display_names","save_summaries")

# GRAPH STRUCTURE

'''Any network can be constructed though a combination of serial (linear_struct) and parallel (branch_struct), but it
might be tedious to define them all seperately. Instead we wish to define blocks and how they are connected to one another.
Since we are working strictly with FFNN, we have one single input from which the shape of all the subsequent blocks are determined.
We also need a check to tell us when the cumulative pooling factors is unworkable.


'''

def graph_structure(nodes,edges):
    pass