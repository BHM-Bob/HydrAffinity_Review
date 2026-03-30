import os
from pymol import cmd
import tempfile
import torch
import numpy as np
from PIL import Image


def setup_pymol_scene(molecule_name):
    """设置PyMOL场景的基本参数"""
    cmd.zoom(complete=1)
    cmd.bg_color('white')
    cmd.set('ray_trace_mode', 0)
    cmd.set('ray_trace_frames', 0)


def set_molecule_representation(molecule_name, rep_type):
    """根据表示类型设置分子的显示方式"""
    cmd.hide('everything', molecule_name)
    
    if rep_type == 'ball_stick':
        cmd.show('spheres', molecule_name)
        cmd.show('sticks', molecule_name)
        cmd.set('sphere_scale', 0.3)
        cmd.set('stick_radius', 0.2)
    elif rep_type == 'stick':
        cmd.show('sticks', molecule_name)
        cmd.set('stick_radius', 0.15)
    elif rep_type == 'surface':
        cmd.show('surface', molecule_name)
        cmd.set('surface_color', 'skyblue')
        cmd.set('transparency', 0.3)


def image_to_tensor(img_path):
    """
    将图像文件转换为PyTorch tensor
    """
    with Image.open(img_path).convert('RGB') as img:
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1) / 255.0
    return img_tensor


def generate_rotated_views_tensor(pdb_file_path, image_size=512,
                                  z_angles=None, x_angles=None,
                                  representations=('ball_stick', 'stick', 'surface'),
                                  save_to_file=False, output_dir=None):
    """
    生成分子的多视角图片tensor，支持不同的表示方式和旋转角度
    
    参数:
    - pdb_file_path: PDB文件路径
    - image_size: 图片尺寸
    - z_angles: 绕z轴旋转的角度列表
    - x_angles: 绕x轴旋转的角度列表
    - representations: 要生成的分子表示方式列表
    - save_to_file: 是否同时保存为文件
    - output_dir: 文件保存目录，当save_to_file=True时使用
    
    返回:
    - 包含所有生成图片tensor的字典，格式为{视角类型: [tensor1, tensor2, ...]}
    - 如果save_to_file=True，同时返回包含文件路径的字典
    """
    # 默认参数设置
    if z_angles is None:
        z_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    if x_angles is None:
        x_angles = [0, 30, 60, 90, 120, 150, 180, 210]
    
    # 如果需要保存文件，创建输出目录
    if save_to_file and output_dir is None:
        output_dir = tempfile.mkdtemp()
    elif save_to_file:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        cmd.reinitialize()
        # 加载分子
        molecule_name = os.path.basename(pdb_file_path).replace('.pdb', '')
        cmd.load(pdb_file_path, molecule_name)
        setup_pymol_scene(molecule_name)
        
        # 初始化结果字典
        result_tensors = {}
        if save_to_file:
            result_paths = {}
        
        # 为每种表示方式生成不同视角
        for rep_type in representations:
            set_molecule_representation(molecule_name, rep_type)
            
            # 生成绕z轴旋转的视图
            z_key = f'z_rotation_{rep_type}'
            result_tensors[z_key] = []
            if save_to_file:
                result_paths[z_key] = []
            
            for i, angle in enumerate(z_angles):
                cmd.orient(molecule_name)
                cmd.turn('z', angle)
                cmd.zoom(complete=1, buffer=1)
                cmd.viewport(image_size, image_size)
                
                # 使用临时文件保存图像
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                    temp_img_path = temp_img.name
                    
                # 保存图像到临时文件
                cmd.png(temp_img_path, ray=1, width=image_size, height=image_size)
                
                # 将图像转换为tensor
                img_tensor = image_to_tensor(temp_img_path)
                result_tensors[z_key].append(img_tensor)
                
                # 如果需要保存文件
                if save_to_file:
                    img_filename = f"{molecule_name}_z_{rep_type}_{i+1}_{angle}deg.png"
                    img_path = os.path.join(output_dir, img_filename)
                    # 复制临时文件到目标位置
                    os.system(f'cp {temp_img_path} {img_path}')
                    result_paths[z_key].append(img_path)
                
                # 删除临时文件
                os.unlink(temp_img_path)
            
            # 生成绕x轴旋转的视图
            x_key = f'x_rotation_{rep_type}'
            result_tensors[x_key] = []
            if save_to_file:
                result_paths[x_key] = []
            
            for i, angle in enumerate(x_angles):
                cmd.orient(molecule_name)
                cmd.turn('x', angle)
                cmd.zoom(complete=1, buffer=1)
                cmd.viewport(image_size, image_size)
                
                # 使用临时文件保存图像
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img:
                    temp_img_path = temp_img.name
                    
                # 保存图像到临时文件
                cmd.png(temp_img_path, ray=1, width=image_size, height=image_size)
                
                # 将图像转换为tensor
                img_tensor = image_to_tensor(temp_img_path)
                result_tensors[x_key].append(img_tensor)
                
                # 如果需要保存文件
                if save_to_file:
                    img_filename = f"{molecule_name}_x_{rep_type}_{i+1}_{angle}deg.png"
                    img_path = os.path.join(output_dir, img_filename)
                    # 复制临时文件到目标位置
                    os.system(f'cp {temp_img_path} {img_path}')
                    result_paths[x_key].append(img_path)
                
                # 删除临时文件
                os.unlink(temp_img_path)
        
        # 清理
        cmd.delete('all')
        
        if save_to_file:
            print(f"所有视角图片已生成到: {output_dir}")
            return result_tensors, result_paths
        else:
            return result_tensors
        
    except Exception as e:
        print(f"生成分子视角tensor时出错: {e}")
        return None
    finally:
        # 保持与原代码一致，不退出PyMOL
        pass


# 保持向后兼容的函数
def generate_rotated_views(pdb_file_path, output_dir=None, image_size=512,
                          z_angles=None, x_angles=None,
                          representations=('ball_stick', 'stick', 'surface')):
    """
    向后兼容的函数，调用新的tensor生成函数并只返回路径
    """
    _, result_paths = generate_rotated_views_tensor(
        pdb_file_path, image_size=image_size, 
        z_angles=z_angles, x_angles=x_angles,
        representations=representations,
        save_to_file=True, output_dir=output_dir
    )
    return result_paths


def load_image_model(model_name: str = 'resnet50'):
    """
    加载适合10G显存的高性能图像模型(resnet50, ViT)
    根据不同模型使用不同的加载方式，选择轻量级但性能好的模型
    
    参数:
    - model_name: 模型名称，可选 'resnet50' 或 'vit'
    
    返回:
    - model: 加载的图像模型
    - preprocessor: 图像预处理工具
    - device: 使用的设备（GPU或CPU）
    """
    try:
        # 检查是否有可用的GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载图像模型到 {device}...")
        
        if model_name.lower() == 'vit':
            # 使用 torchvision 直接加载 ViT 模型
            from torchvision import models
            model = models.vit_b_16(pretrained=True)
        else:  # 默认使用 ResNet50
            # 使用torch hub加载ResNet50 - 性能好且内存占用适中
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        model = model.to(device)
        model.eval()  # 设置为评估模式
            
        def preprocessor(images):
            """预处理图像批次"""
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            
            # 处理单张图像或图像批次
            if isinstance(images, torch.Tensor) and images.ndim == 3:
                return transform(images).unsqueeze(0)  # 添加批次维度
            elif isinstance(images, list):
                return torch.stack([transform(img) for img in images])
            return transform(images)
        
        print(f"{model_name} 图像模型加载完成")
        return model, preprocessor, device
    
    except Exception as e:
        print(f"加载图像模型时出错: {e}")
        return None, None, None


def process_image_tensors(img_tensors_dict, model, preprocessor, device, batch_size=4):
    """
    使用图像模型处理tensor字典，返回隐藏状态
    
    参数:
    - img_tensors_dict: 包含图像tensor的字典 {视角类型: [tensor1, tensor2, ...]}
    - model: 加载的图像模型
    - preprocessor: 图像预处理函数
    - device: 使用的设备
    - batch_size: 批处理大小
    
    返回:
    - 包含隐藏状态的字典 {视角类型: [hidden_state1, hidden_state2, ...]}
    """
    hidden_states_dict = {}
    
    # 为模型添加钩子以获取中间特征
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # 根据模型类型选择合适的钩子位置
    is_vit = False
    # 检查是否是 ViT 模型
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        is_vit = True
        # 对于 ViT，我们获取最后一层编码器的输出
        last_encoder_layer = model.encoder.layers[-1]
        last_encoder_layer.register_forward_hook(get_features('last_encoder_layer'))
    elif hasattr(model, 'avgpool'):
        # 对于 ResNet 等 CNN 模型
        model.avgpool.register_forward_hook(get_features('avgpool'))
    else:
        # 通用情况：尝试获取最后几个模块中的一个
        for name, module in list(model.named_modules())[-5:-1]:  # 尝试最后几个模块
            if hasattr(module, 'register_forward_hook'):
                module.register_forward_hook(get_features(name))
                break
    
    # 处理每种视角类型的图像
    for view_type, tensors in img_tensors_dict.items():
        hidden_states = []
        
        for i in range(0, len(tensors), batch_size):
            batch = tensors[i:i+batch_size]
            
            batch_preprocessed = preprocessor(batch).to(device)
            
            with torch.no_grad():
                # 对于 ViT，我们需要特别处理特征提取
                if is_vit:
                    # 先获取输出
                    outputs = model(batch_preprocessed)
                    # 然后获取钩子捕获的特征
                    if 'last_encoder_layer' in features:
                        # ViT 最后一层的输出形状通常是 [batch_size, num_patches+1, embedding_dim]
                        # 我们使用 cls token 的特征 (第一个token) 或对所有token取平均
                        vit_features = features['last_encoder_layer']
                        # 获取 cls token 特征 (通常是序列的第一个元素)
                        cls_features = vit_features[:, 0]
                        
                        for feat in cls_features.cpu():
                            hidden_states.append(feat)
                else:
                    # 对于 CNN 模型，正常前向传播并获取钩子特征
                    _ = model(batch_preprocessed)
                    
                    if features:
                        # 获取最后保存的特征
                        last_feature_key = list(features.keys())[-1]
                        batch_features = features[last_feature_key].cpu()
                        
                        # 展平特征并添加到结果列表
                        for feat in batch_features:
                            hidden_states.append(feat.view(-1))  # 展平为向量
        
        hidden_states_dict[view_type] = hidden_states
    
    return hidden_states_dict


# 保存张量为图片的函数
def tensor_to_image(tensor, output_path):
    """
    将PyTorch张量转换为图像并保存
    
    参数:
    - tensor: 形状为 [C, H, W] 的PyTorch张量，值范围应该是 [0, 1]
    - output_path: 输出图片的保存路径
    """
    # 确保张量在CPU上且为numpy格式
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    # 将张量转换为numpy数组，并调整形状为 [H, W, C]
    # 如果张量值范围是 [0, 1]，需要乘以255转换为 [0, 255]
    img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    # 创建PIL图像并保存
    img = Image.fromarray(img_np)
    img.save(output_path)
    print(f"图像已保存到: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 示例PDB文件（您需要替换为实际的PDB文件路径）
    pdb_file = "../EHIGN_dataset/PDBbind_2019/pdb/1a0q/1a0q_ligand.pdb"  # 替换为您的PDB文件路径
    
    if os.path.exists(pdb_file):
        # 仅生成tensor，不保存文件
        img_tensors = generate_rotated_views_tensor(
            pdb_file, image_size=224,  # 为了与图像模型匹配，使用224x224
            representations=('ball_stick', 'stick', 'surface')
        )
        
        # 加载适合10G显存的图片模型
        model, preprocessor, device = load_image_model('vit')
        
        if model is not None:
            # 处理图像tensor获取隐藏状态
            hidden_states = process_image_tensors(
                img_tensors, 
                model, 
                preprocessor, 
                device, 
                batch_size=4  # 可以根据实际显存调整批次大小
            )
            
            if hidden_states:
                print("\n处理后的隐藏状态统计:")
                for view_type, states in hidden_states.items():
                    print(f"{view_type}: {len(states)} 个隐藏状态, 形状: {states[0].shape}")
        
    else:
        print(f"PDB文件不存在: {pdb_file}")
        print("请将示例PDB文件保存为 'example.pdb' 或修改文件路径")