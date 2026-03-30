import torch

def check_available_sdp_backends():
    """
    检测机器上可用的SDP后端
    """
    print("检测可用的SDP后端...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU计算能力: {torch.cuda.get_device_capability(0)}")
    
    try:
        # 尝试导入新的API
        from torch.nn.attention import SDPBackend, sdpa_kernel
        
        print("\n使用torch.nn.attention API:")
        print(f"可用的后端类型: {[b for b in dir(SDPBackend) if not b.startswith('_')]}")
        
        # 测试每个后端是否可用
        backends = {
            'FLASH_ATTENTION': SDPBackend.FLASH_ATTENTION,
            'EFFICIENT_ATTENTION': SDPBackend.EFFICIENT_ATTENTION,
            'MATH': SDPBackend.MATH,
            'CUDNN_ATTENTION': SDPBackend.CUDNN_ATTENTION
        }
        
        available_backends = []
        for name, backend in backends.items():
            try:
                # 创建小的测试张量
                q = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                k = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                v = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                
                # 尝试使用特定后端
                with sdpa_kernel(backend):
                    output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                available_backends.append((name, True))
                print(f"✅ {name} 后端可用")
            except Exception as e:
                available_backends.append((name, False))
                print(f"❌ {name} 后端不可用: {str(e)}")
        
        return available_backends
    except ImportError:
        print("\n使用旧的torch.backends.cuda API:")
        # 尝试使用旧的API
        try:
            from torch.backends.cuda import sdp_kernel as sdpa_kernel
            
            # 测试不同的后端配置
            configurations = [
                ('FLASH', {'enable_flash': True, 'enable_math': False, 'enable_mem_efficient': False}),
                ('MEMORY_EFFICIENT', {'enable_flash': False, 'enable_math': False, 'enable_mem_efficient': True}),
                ('MATH', {'enable_flash': False, 'enable_math': True, 'enable_mem_efficient': False})
            ]
            
            available_backends = []
            for name, config in configurations:
                try:
                    # 创建小的测试张量
                    q = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                    k = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                    v = torch.randn(1, 1, 16, 16).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 16, 16)
                    
                    # 尝试使用特定配置
                    with sdpa_kernel(**config):
                        output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
                    available_backends.append((name, True))
                    print(f"✅ {name} 后端可用")
                except Exception as e:
                    available_backends.append((name, False))
                    print(f"❌ {name} 后端不可用: {str(e)}")
            
            return available_backends
        except Exception as e:
            print(f"无法使用任何SDP API: {str(e)}")
            return []

if __name__ == "__main__":
    available_backends = check_available_sdp_backends()
    print("\n检测完成!")
    if available_backends:
        print("\n可用的后端摘要:")
        for name, available in available_backends:
            status = "✅ 可用" if available else "❌ 不可用"
            print(f"{name}: {status}")
    else:
        print("没有检测到可用的SDP后端")