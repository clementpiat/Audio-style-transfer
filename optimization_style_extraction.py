
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda




def getIterateFunctionGram_notfinished(model,layerDict,inputTensor,Image):
    
    for layerName in layerDict.keys():
        if('conv' in layerName):
            layerOutput = layerDict[layerName].output
            filtersNumber = tf.shape(layerOutput)[-1]
            
            for i in range(filtersNumber):
                for i in range(filtersNumber):
                    #calcul of Gijl
                    Fi=layerOutput[:,:,:,i]
                    Fj=layerOutput[:,:,:,j]
                    Gijl = tf.math.reduce_sum(tf.multiply(Fi,Fj))

                    
def getTargetedActivation(model,layerDict,layerName,inputTensor,Image):
    layerOutput = layerDict[layerName].output
    iterate = K.function([inputTensor], [layerOutput])
    return iterate([Image])


def getIterateFunctionContent(model,layerDict, layerName, inputTensor,Image):
    
    layerOutput = layerDict[layerName].output[0,:,:,:]
    arrayTarget = np.squeeze(getTargetedActivation(model,layerDict,layerName,inputTensor,Image)[0])
    layerTarget = tf.convert_to_tensor(arrayTarget)

    difference=tf.math.subtract(layerOutput,layerTarget)

    loss=K.sum(K.square(difference))/(128*259*64)

    grads = K.gradients(loss, inputTensor)[0]

    iterate = K.function([inputTensor], [loss, grads])
    
    return iterate


def getIterateFunctionGram(model,layerDict,inputTensor,Image,w=[0.2,0,0.2,0,0.2,0,0,0.2,0,0,0.2,0,0]):
    
    loss=tf.cast(0,tf.float64)
    index=0
    for layerName in layerDict.keys():
        if('conv' in layerName):
                if(w[index]!=0):
                    layerOutput = layerDict[layerName].output
                    targetOutput = tf.convert_to_tensor(np.squeeze(getTargetedActivation(model,layerDict,layerName,inputTensor,Image)))

                    Nl = tf.shape(layerOutput)[-1]
                    Ml = tf.shape(layerOutput)[-2]*tf.shape(layerOutput)[-3]
                    Gl = tf.tensordot(layerOutput,layerOutput,[[0,1],[0,1]])
                    Al = tf.tensordot(targetOutput,targetOutput,[[0,1],[0,1]])

                    difference=tf.math.subtract(layerOutput,targetOutput)

                    wl=w[index]
                    square=K.square(difference)
                    coeff=1/(2*Nl*Ml)
                    s=K.sum(square)*wl
                    s=tf.cast(s, tf.float64)
                    s=s*coeff

                    loss = tf.math.add(loss,s)
                index+=1
            

    grads = K.gradients(loss, inputTensor)[0]

    iterate = K.function([inputTensor],[loss,grads])
    return(iterate)
 
def getIterateFunctionSumNope(model,layerDict,inputTensor,ImageContent,ImageStyle,a,b,w=[0.2,0,0.2,0,0.2,0,0,0.2,0,0,0.2,0,0]):
    
    iterateContent = getIterateFunctionContent(model,layerDict,"block1_conv1",inputTensor,ImageContent)
    iterateStyle = getIterateFunctionGram(model,layerDict,inputTensor,ImageStyle,w=[0.2,0,0.2,0,0.2,0,0,0.2,0,0,0.2,0,0])
    
    lossContent=iterateContent([inputTensor])[0]
    lossStyle = iterateStyle([inputTensor])[0]
    
    loss = a*lossContent + b*lossStyle
    grads = K.gradients(loss,inputTensor)[0]
    
    iterate = K.function([inputTensor],[loss,grads])
    return(iterate)

def getIterateFunctionSum(model,layerDict,inputTensor,ImageContent,ImageStyle,a,b,w=[0.2,0,0.2,0,0.2,0,0,0.2,0,0,0.2,0,0]):
    
    ##lossStyle
    lossStyle=tf.cast(0,tf.float64)
    index=0
    for layerName in layerDict.keys():
        if('conv' in layerName):
                if(w[index]!=0):
                    layerOutput = layerDict[layerName].output
                    targetOutput = tf.convert_to_tensor(np.squeeze(getTargetedActivation(model,layerDict,layerName,inputTensor,ImageStyle)))

                    Nl = tf.shape(layerOutput)[-1]
                    Ml = tf.shape(layerOutput)[-2]*tf.shape(layerOutput)[-3]
                    Gl = tf.tensordot(layerOutput,layerOutput,[[0,1],[0,1]])
                    Al = tf.tensordot(targetOutput,targetOutput,[[0,1],[0,1]])

                    difference=tf.math.subtract(layerOutput,targetOutput)

                    wl=w[index]
                    square=K.square(difference)
                    coeff=1/(2*Nl*Ml)
                    s=K.sum(square)*wl
                    s=tf.cast(s, tf.float64)
                    s=s*coeff

                    lossStyle = tf.math.add(lossStyle,s)
                index+=1
    ##lossContent
    layerOutput = layerDict["block1_conv1"].output[0,:,:,:]
    arrayTarget = np.squeeze(getTargetedActivation(model,layerDict,"block1_conv1",inputTensor,ImageContent)[0])
    layerTarget = tf.convert_to_tensor(arrayTarget)
    difference=tf.math.subtract(layerOutput,layerTarget)
    lossContent=K.sum(K.square(difference))/(128*259*64)
    
    #loss
    loss = tf.math.add(tf.cast(a*lossContent,tf.float64) , b*lossStyle)
    grads = K.gradients(loss,inputTensor)[0]
    
    iterate = K.function([inputTensor],[loss,grads,lossContent,lossStyle])
    return(iterate)
    
    
    
    
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
            if(longueur==4):
                lossValue, gradsValue,loss_content,loss_style = liste
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
            if(i%10==0):
                print(i,lossValue,np.max(abs((-1*lr)*mi / (np.sqrt(vi)+eps))))
                if(longueur==4):
                    print(loss_content,loss_style)
            
    return(lossValue)
                    
                        