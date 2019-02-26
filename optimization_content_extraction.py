
import numpy as np
import tensorflow as tf
from keras import backend as K

def getTargetedActivation(model,layerDict,layerName,inputTensor,Image):
    layerOutput = layerDict[layerName].output
    iterate = K.function([inputTensor], [layerOutput])
    return iterate([Image])

from keras.layers import Input, Lambda

    
def getIterateFunctionStyle(model,layerDict, layerName, inputTensor,Image,a,b,c,styleIndex):
    
    layerOutput = layerDict[layerName].output[0,:,:,:]
    arrayTarget = np.squeeze(getTargetedActivation(model,layerDict,layerName,inputTensor,Image)[0])
    print("max target",np.max(abs(arrayTarget)))
    layerTarget = tf.convert_to_tensor(arrayTarget)
    print(layerOutput)
    print(layerTarget)
    print(layerOutput.shape)
    print(layerTarget.shape)
    #subtract_layer = Lambda(lambda inputs: inputs[0] - inputs[1],output_shape=lambda shapes: shapes[0])
    #subtract = K.function([inputTensor],[subtract_layer([inputTensor,layerTarget])])
    #difference = subtract[layerOutput]
    difference=tf.math.subtract(layerOutput,layerTarget)
    #difference=K.bias_add(layerOutput[:,:,:,:],-1*getTargetedActivation(model,layerDict,layerName,inputTensor,Image))
    loss_content=K.sum(K.square(difference))/(128*259*64)
    
    loss_style= - tf.math.log(K.mean(model.output[:,styleIndex]))
    
    Regularization_term = K.mean(K.square(inputTensor))
    
    loss=a*loss_content+b*loss_style+c*Regularization_term
    #loss=loss_content
    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, inputTensor)[0]


    
    iterate = K.function([inputTensor], [loss, grads,loss_content,loss_style,Regularization_term])
    
    return iterate
    
def getIterateFunction(model,layerDict, layerName, inputTensor,Image):
    
    layerOutput = layerDict[layerName].output[0,:,:,:]
    arrayTarget = np.squeeze(getTargetedActivation(model,layerDict,layerName,inputTensor,Image)[0])
    layerTarget = tf.convert_to_tensor(arrayTarget)

    difference=tf.math.subtract(layerOutput,layerTarget)

    loss=K.sum(K.square(difference))/(128*259*64)

    grads = K.gradients(loss, inputTensor)[0]

    iterate = K.function([inputTensor], [loss, grads])
    
    return iterate


##first attempt : kind of Adagrad
def gradientDescent(iterate, inputImgData, step):
    k=5
    lossValue=2000
    for i in range(1,300):
        if(lossValue/512>=3):
            lossValue, gradsValue = iterate([inputImgData])
            gradsValue /= (K.sqrt(K.max(K.square(gradsValue))) + 1e-10)/20
            print(lossValue,np.max(abs(gradsValue))*step/(1+k*i))
            inputImgData+=gradsValue*step/(1+k*i)*(-1)
    return lossValue

def SGD_Nesterov(iterate,img,lr0,momentum,early_stopping,factor=0.5,epochs=20,niter=200):
    #initialization
    lossValue=early_stopping+1
    v=np.zeros(img.shape)
    lr=lr0
    #update of img
    for i in range(1,niter):
        if(lossValue>early_stopping):
            lossValue, gradsValue = iterate([img])
            #step decay for learning rate
            lr=lr*(factor**(i%epochs==0))
            #Nesterov momentum + SGD
            v_prec=v.copy()
            v=momentum*v-lr*gradsValue
            img+= (-1*momentum)*v_prec + (1+momentum)*v
            print(i,lossValue,np.max(abs(v)),np.max(abs(gradsValue)))
    
    return(lossValue)
    
def Adam(iterate,img,lr0,early_stopping,factor,epochs,niter,beta1=0.9,beta2=0.999,eps=1e-8):
    #initialization
    lossValue=early_stopping+1
    v=np.zeros(img.shape)
    m=np.zeros(img.shape)
    lr=lr0

    #update of img
    for i in range(1,niter):
        if(lossValue>early_stopping):
            liste=iterate([img]) 
            longueur=len(liste)
            if(longueur==5):
                lossValue, gradsValue,loss_content,loss_style,reg = liste
            elif(longueur==2):
                lossValue,gradsValue=liste
            #step decay for learning rate
            lr=lr*(factor**(i%epochs==0))
            #ADAM
            m=beta1*m + (1-beta1)*gradsValue
            mi=m/(1-beta1**i)
            v=beta2*v + (1-beta2)*(gradsValue**2)
            vi=v/(1-beta2**i)
            img+= (-1*lr)*mi / (np.sqrt(vi)+eps)
            if(i%1==0):
                print(i,lossValue,np.max(abs((-1*lr)*mi / (np.sqrt(vi)+eps))))
                if(longueur==5):
                    print(loss_content,loss_style,reg)
            
    return(lossValue)