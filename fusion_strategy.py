import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils


EPSILON = 1e-5


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2, p_type):
    # avg, max, nuclear
    f_channel_0 = channel_fusion(tensor1[0], tensor2[0],  p_type)
    # f_channel_1 = channel_fusion(tensor1[1], tensor2[1], p_type)
    # f_channel_2 = channel_fusion(tensor1[2], tensor2[2], p_type)

    f_spatial_0 = spatial_fusion(tensor1[0], tensor2[0])
    f_spatial_1 = spatial_fusion(tensor1[1], tensor2[1])
    f_spatial_2 = spatial_fusion(tensor1[2], tensor2[2])

    tensor_f_0 = (f_channel_0 + f_spatial_0) / 2
    # tensor_f_1 = (f_channel_1 + f_spatial_1) / 2
    # tensor_f_2 = (f_channel_2 + f_spatial_2) / 2
    tensor_f_1 = f_spatial_1
    tensor_f_2 = f_spatial_2
    return (tensor_f_0,tensor_f_1,tensor_f_2)


# select channel
def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)  #为什么加EPSILON?
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type is 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type is 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type is 'attention_nuclear':
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):   #
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)         #通道维度上的平均数
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)        #通道维度上的和
    return spatial


# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors



