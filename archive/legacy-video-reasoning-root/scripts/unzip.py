import os
import sys
import tarfile
from tqdm import tqdm

# 设置输出编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 这里是你的下载路径
SOURCE_DIR = r"D:\SA\URIS\dataset"

def extract_all_tars(root_dir):
    # 遍历所有文件
    tar_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_files.append(os.path.join(root, file))
    
    print(f"找到 {len(tar_files)} 个压缩包，准备解压...")
    sys.stdout.flush()

    for tar_path in tar_files:
        # 获取当前压缩包所在的文件夹路径
        extract_path = os.path.dirname(tar_path)
        print(f"正在解压: {os.path.basename(tar_path)} -> {extract_path}")
        sys.stdout.flush()
        
        try:
            with tarfile.open(tar_path, "r:gz") as tar:
                # 获取所有成员文件，用于进度条
                members = tar.getmembers()
                # 过滤出视频文件（避免解压垃圾文件）
                video_members = [m for m in members if m.name.endswith('.mp4')]
                
                if not video_members:
                    print("  ⚠️ 包内没有找到 MP4 文件，跳过。")
                    continue
                
                # 带进度条解压
                for member in tqdm(video_members, desc="  提取中", unit="file"):
                    tar.extract(member, path=extract_path, filter='data')
                    
            print("  ✅ 解压完成")
            sys.stdout.flush()
            
            # (可选) 解压完删除原压缩包以节省空间，如果你硬盘够大可以注释掉下面这行
            # os.remove(tar_path) 
            
        except Exception as e:
            print(f"  ❌ 解压出错: {e}")

if __name__ == "__main__":
    extract_all_tars(SOURCE_DIR)
    print("\n所有操作结束！请检查文件夹内的 .mp4 文件。")
    sys.stdout.flush()