import datetime,os,shutil
def copy_folder_with_exceptions(src_folder, dest_folder, exclude_folders):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    else:
        return
    print(src_folder)
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dest_item = os.path.join(dest_folder, item)

        if any(exclude_folder in src_item for exclude_folder in exclude_folders):
            continue
        print(src_item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dest_item, symlinks=True, ignore=None)
        else:
            shutil.copy2(src_item, dest_item)
            
def get_formatted_date():
    """
    Returns:
        str: 当前日期的 YY-MM-DD 格式字符串。
    """
    current_date = datetime.datetime.now()  # 获取当前日期时间对象
    formatted_date = current_date.strftime('%y-%m-%d')  # 将日期时间格式化为 YY-MM-DD 字符串
    return formatted_date
    
expname = "latent_diffusion_ae_module"
cur_time = get_formatted_date()

# Backup code
log_folder = f'code_version/{cur_time}_{expname}/'
source_folder = "./LatentDiffusion"
dest_folder_code = os.path.join(log_folder,source_folder)
excluded_folders = ["checkpoints","sample","logs", "__pycache__"] #".git", 
copy_folder_with_exceptions(source_folder, dest_folder_code, excluded_folders)
print("code copy done. ")

