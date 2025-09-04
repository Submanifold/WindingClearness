import torch.optim as optim
import torch.utils.data
import argparse
from utils.visualize import *
from datasets.test_data_pc import PointCloud
torch.set_default_dtype(torch.float32)
from torch.autograd import Variable
import random


def normalize_shift_scale(pc):
    pc = pc.squeeze()
    pc_max, _ = pc.max(dim=0, keepdim=True)  # (1, 3)
    pc_min, _ = pc.min(dim=0, keepdim=True)  # (1, 3)
    shift = ((pc_min + pc_max) / 2).view(1, 1, 3)
    scale = (pc_max - pc_min).max().reshape(1, 1)
    return shift, scale


def sample_bound_points(xmin, xmax, ymin, ymax, zmin, zmax, total_point_num):
    box = trimesh.creation.box(extents=[xmax-xmin, ymax-ymin, zmax-zmin], origin=[(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2])
    bound_pc = trimesh.sample.sample_surface_even(box, count=total_point_num)
    return bound_pc[0]


def get_A(x, y, x_width):
    x_u1 = x.unsqueeze(1)
    y_u2 = y.unsqueeze(2)
    A = x_u1 - y_u2  # [N_query, N_sample, 3]
    A_squared = torch.abs(A) ** 2
    dist = torch.sqrt(torch.clamp(A_squared.sum(-1), min=float(1e-6)))  # [N_query, N_sample], |x^i-y^j|^2
    dist = dist.float()
    # inv_dist = cnp.where(dist > x_width[:, None], 1/dist, 0.) # / 4 / cp.pi
    inv_dist = torch.where(dist.float() > x_width, 1 / (dist + float(1e-6)), 1 / x_width)  # / 4 / cp.pi
    inv_cub_dist = inv_dist ** 3 / (4 * torch.tensor(3.14159265359, dtype=torch.float32))  # [N_query, N_sample]


    A = (A * inv_cub_dist[..., None])  # [N_query, N_sample, 3]
    A = A.transpose(2, 3)
    A = A.reshape(x.size(0), y.size(1), -1)

    return -A


def compute_loss(x, p_origin, prev_x, device, bound_point_torch, opt):

    if prev_x.numel() == 0:
        concatenated_x_prev_x = x
    else:
        concatenated_x_prev_x = torch.cat([x, prev_x], dim=1)
    bound_point_torch_size = bound_point_torch.size()
    x_size = x.size()
    concatenated_x_prev_x_size = concatenated_x_prev_x.size()
    batch_size = x_size[0]
    num_point = concatenated_x_prev_x_size[1]
    dim_point_cloud = x_size[2]
    b = torch.ones((batch_size, 1, concatenated_x_prev_x_size[1])) * 0.5
    b2 = torch.ones((batch_size, 1, bound_point_torch_size[1])) * 0.0
    b = b.to(device)
    b2 = b2.to(device)
    bT = b.transpose(1, 2)
    b2T = b2.transpose(1, 2)
    x_width = float(opt.wx)
    x_width = torch.tensor(x_width, dtype=torch.float32)
    x_width = x_width.to(device)
    A = get_A(concatenated_x_prev_x, concatenated_x_prev_x, x_width)

    lambda_regu = float(opt.alpha)
    At = A.transpose(1, 2)

    AtA = At @ A
    diagonalAtA = torch.diag(AtA.squeeze(0))
    diagonalAtA_matrix = torch.diag_embed(diagonalAtA)
    lhs = AtA + lambda_regu * diagonalAtA_matrix.unsqueeze(0)
    rhs = At @ bT
    lambda_j = opt.lambda_j
    A2 = get_A(concatenated_x_prev_x, bound_point_torch, x_width)
    A2t = A2.transpose(1, 2)
    A2tA2 = A2t @ A2
    diagonalA2tA2 = torch.diag(A2tA2.squeeze(0))
    diagonalA2tA2_matrix = torch.diag_embed(diagonalA2tA2)
    lhs = lhs + lambda_j * A2tA2 + lambda_regu * lambda_j * diagonalA2tA2_matrix.unsqueeze(0)

    mu = torch.linalg.solve(lhs, rhs)
    new_a_mu = A @ mu
    loss = []
    lambda_k = opt.lambda_k

    pgr_loss = torch.nn.functional.mse_loss(new_a_mu, bT)

    pgr_loss = pgr_loss + lambda_regu * torch.bmm(mu.transpose(1, 2), torch.bmm(diagonalAtA_matrix.unsqueeze(0), mu)) / num_point
    loss.append(pgr_loss)
    distance_loss = torch.nn.functional.mse_loss(x, p_origin) * lambda_k * dim_point_cloud # 加入权重
    loss.append(distance_loss)


    new_a2_mu = A2 @ mu
    pgr_loss2 = torch.nn.functional.mse_loss(new_a2_mu, b2T) * lambda_j * 2

    pgr_loss2 = pgr_loss2 + lambda_j * lambda_regu * torch.bmm(mu.transpose(1, 2), torch.bmm(diagonalA2tA2_matrix.unsqueeze(0), mu)) / num_point
    loss.append(pgr_loss2)

    print("distance_loss: " + str(distance_loss.item()))
    print("pgr_loss: " + str(pgr_loss.item()))
    if opt.regularize_bounding_box:
        print("pgr_loss2: " + str(pgr_loss2.item()))


    return loss



def get_dataset(dataroot):
    tr_dataset = PointCloud(root_dir=dataroot)
    return tr_dataset

def get_dataloader(opt, train_dataset):
    train_sampler = None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler,
                                                   shuffle=train_sampler is None, num_workers=int(opt.workers),
                                                   drop_last=True)
    return train_dataloader, train_sampler


def train(opt):
    gpu_idx = 0
    device = torch.device("cuda:%d" % gpu_idx)

    train_dataset = get_dataset(opt.dataroot)
    dataloader, train_sampler = get_dataloader(opt, train_dataset)
    for data_iter, data in enumerate(dataloader):
        x0 = data['points']
        second_dim = x0.shape[1]
        random_index = torch.randperm(second_dim)


        x0_shuffled = x0[:, random_index, :]
        x0 = x0_shuffled
        shift, scale = normalize_shift_scale(x0)
        x0 = (x0 - shift) / scale
        total_point_num = x0.cpu().shape[1]

        file_name = data['file_name']
        folder_path = f"./optimization_based_temp/{file_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        loss_history_file_path = f"./output/{opt.result_label}/{file_name}_loss.txt"

        with open(loss_history_file_path, 'w') as f:
            f.write('')

            
        prev_x = x0.clone().detach().requires_grad_(False)
        x0 = x0.cuda().float()

        bound_coord = 0.6
        bound_point_numpy = sample_bound_points(-bound_coord, bound_coord, -bound_coord, bound_coord, -bound_coord,
                                                bound_coord, opt.bound_point_count)
        bound_point_numpy = np.repeat(bound_point_numpy[np.newaxis, :, :], 1, axis=0)
        bound_point_torch = torch.from_numpy(bound_point_numpy).to(dtype=torch.float32)
        bound_point_torch = bound_point_torch.to(device)

        patch_size = opt.patch_size


        index_offset = 0
        while True:
            if (index_offset >= total_point_num):
                break

            if index_offset == 0:
                num_points = 5000
            else:
                num_points = 2500


            indices = list(range(index_offset, index_offset + num_points))

            x = x0[:, indices, :].clone().detach().requires_grad_(True)
            if opt.use_adam:
                optimizer = optim.Adam([{'params': x, 'requires_grad': True}], lr=opt.lr)
            else:
                optimizer = optim.SGD([{'params': x, 'requires_grad': True}], lr=opt.lr, momentum=0.9)

            if index_offset == 0:
                prev_x_for_patch = torch.empty(0, 0, 0)
            else:
                random_indices = random.sample(range(index_offset), opt.previous_x_pool_size)
                prev_x_for_patch = prev_x[:, random_indices, :].clone().detach().requires_grad_(False)
            prev_x_for_patch = prev_x_for_patch.cuda()
            x = x.cuda()
            start_epoch = 0
            p_origin = x.clone().detach().requires_grad_(False)
            for epoch in range(start_epoch, opt.niter):
                optimizer.zero_grad()
                loss = compute_loss(x, p_origin, prev_x_for_patch, device, bound_point_torch, opt)
                loss_total = sum(loss)
                loss_total.backward()
                optimizer.step()

                current_loss = loss_total.item()



            index_offset += num_points
            prev_x[:, indices, :] = x.detach().cpu()

            print(f"data: {data_iter} epoch: {epoch} total_loss: {current_loss:.6f}")



        prev_x = prev_x[:, :index_offset, :]
        y = prev_x * scale + shift
        np.savetxt(f"./output/{opt.result_label}/{file_name}.xyz", y.numpy().squeeze(), delimiter=' ',
                   fmt='%.6f')

def main():
    opt = parse_args()
    train(opt)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./input/3D_multi_batch')
    parser.add_argument('--result_label', default='3D_multi_batch')

    parser.add_argument('--workers', type=int, default=16, help='workers')
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')

    # 新增参数
    parser.add_argument('--bound_point_count', type=int, default=5000, help='Number of points to sample from bounding box')

    parser.add_argument('--nc', default=3)

    # params
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate for E, default=0.0002')
    parser.add_argument('--decay', type=float, default=0, help='weight decay for EBM')
    parser.add_argument('--grad_clip', type=float, default=1e2, help='grad_clip for EBM')

    parser.add_argument('--lambda_j', type=float, default=5.0, help='lambda_j')
    parser.add_argument('--lambda_k', type=float, default=10.0, help='lambda_k')
    parser.add_argument('--wx', type=float, default=0.04, help='wx')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')

    parser.add_argument('--use_adam', type=bool, default=True, help='use adam')
    parser.add_argument('--add_regu_in_loss', type=bool, default=True, help='Add regularization in loss')
    parser.add_argument('--regularize_bounding_box', type=bool, default=True, help='regularize bounding box')

    parser.add_argument('--batch_size_initial', type=int, default=10000, help='Initial batch size for all points')
    parser.add_argument('--patch_size', type=int, default=5000, help='Size of each patch')
    parser.add_argument('--previous_x_pool_size', type=int, default=2500, help='Size of prev_x pool for each patch')

    '''distributed'''
    parser.add_argument('--world_size', default=1, type=int, help='Number of distributed nodes.')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:9991', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--distribution_type', default='single', choices=['multi', 'single', None], help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use. None means using all available GPUs.')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()