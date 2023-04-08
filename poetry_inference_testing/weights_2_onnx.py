import warnings
import onnx 
import onnxruntime as ort
import sys 
import os 
import torch 
import numpy as np 
import argparse
from onnx2torch import convert

# TODO: save or convert torch to onnx 







# TODO: Infer the pretrained model using ONNXruntime
# TODO: Make it viable for all the different models like SSD, YOLO and StyleGANs more. 
# TODO: 




#TODO: make CLi args for the iferencing engine
if __name__ == "__main__":

    print("=>Initializing CLI args")

    parser = argparse.ArgumentParser(description='Args for model inferencing')
    parser.add_argument("--batch_size", type=int, default=28, help="batch size")
    parser.add_argument("--model_name", type=str,default ="YOLO", required=True, help="YOLO/SSD..")  # "16Jan_1_seg"
    args = parser.parse_args()

    onnx_model_path = "/home/gauti/Documents/MLOps/MLOps-AWS-Stack/models/tinyyolov2-7/model/model.onnx"
    
    if args.model_name == "YOLO":

        
        x = torch.ones((1, 3, 416, 416)).cpu()

        # out_torch = torch_ssd_model(x)
        #finding input and output names 

        ssd_model_onnx = onnx.load(onnx_model_path)
        output_nodes = [node.name for node in ssd_model_onnx.graph.output]
        input_nodes = [node.name for node in ssd_model_onnx.graph.input ] 

        print(output_nodes)
        print(input_nodes)


        ort_sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        outputs_ort = ort_sess.run(None, {'image': x.numpy()})
        print(outputs_ort[0].shape)



        
        #TODO: make it viable for torch2onnx
        # ssd_model = onnx.load(onnx_model_path)
        # torch_ssd_model = convert(ssd_model)
        # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
        # utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

       
        # Check the Onnx output against PyTorch
        # print(torch.max(torch.abs(outputs_ort - out_torch.detach().numpy())))
        # print(np.allclose(outputs_ort, out_torch.detach().numpy(), atol=1.e-7))    







