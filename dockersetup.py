import urllib.request
import subprocess
import json


def getComputeCapability(gpu):
    url = 'https://developer.nvidia.com/cuda-gpus'
    with urllib.request.urlopen(url) as r:
        data = r.read().decode('utf-8')
    start = data.find(gpu+'<')
    if start < 0:
        return start
    start += len('gpu')
    v = ''
    b = False
    for i in range(start, len(data)):
        if data[i] == '<':
            b = True
        if b:
            ch = ord(data[i])
            if ch >= ord('0') and ch <= ord('9') or data[i] == '.':
                v += data[i]
            if len(v) == 3:
                break
    return v

def nvidia_smi(query = None):
    command = 'nvidia-smi'
    if query:
        command += f' {query}'
        command = command.split()
    return subprocess.check_output(command)

def getCudaVersion():
    data = nvidia_smi()
    data = data.decode('utf-8').splitlines()
    version = data[2].split(' ')[14]
    if len(version.split('.')) < 3:
        version += '.0'
    return version

def getGraphicsCard():
    # nvidia-smi --query-gpu=gpu_name --format=csv
    data = nvidia_smi('--query-gpu=gpu_name --format=csv')
    return data.decode('utf-8').split(' ')[-1].splitlines()[0]

def getDockerImage(cudaVersion,distribution='ubuntu', type='runtime'):
    # type runtime or devel
    url = f'https://hub.docker.com/v2/repositories/nvidia/cuda/tags/?page_size=25&page=1&name={cudaVersion}-cudnn'
    request = urllib.request.urlopen(url)
    image = ''

    data = json.load(request)
    images = []
    for i in data['results']:
        name = i['name']
        if distribution in name and type in name:
            images.append(name)
    maxv = -1
    for i in images:
        distversion = i.split('-')[-1]
        version = ''
        for j in distversion:
            if j >= '0' and j <= '9' or j == '.':
                version += j
        version = float(version)
        if version > maxv:
            maxv = version
            image = i
    return f'nvidia/cuda:{image}'
def cores_linux():
    return int(subprocess.check_output('nproc'))

def main():
    gpu = getGraphicsCard()
    print('GPU:', gpu)
    computeCap = getComputeCapability(gpu)
    print('Compute capability:', computeCap)
    cudaV = getCudaVersion()
    image = getDockerImage(cudaV)
    print('Docker image:',image)
    with open('.env', 'w') as f:
        f.write(f'GPU_ARCH={computeCap}\n')
        f.write(f'IMAGE={image}\n')
        f.write(f'CORES={cores_linux()}')
if __name__ == '__main__':
    main()