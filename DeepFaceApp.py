from deepface import DeepFace
from PIL import Image
import requests
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.io import read_image
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PotraitFace import PotraitFace
import time


start_time = time.time()
def plot_potraitfaces(potraitfaces_list):
    
    fig, axes = plt.subplots(1, len(potraitfaces_list), figsize=(15, 5))
    col=st.columns(len(potraitfaces_list))
    if len(potraitfaces_list)>1:
        for i in range(len(potraitfaces_list)):
                # Original image
                
            axes[i].imshow(potraitfaces_list[i])
            axes[i].set_title(f'Potrait Image {i+1}')
            axes[i].axis('off')
            with col[i]:
                st.image(potraitfaces_list[i],use_column_width=True)
        plt.tight_layout()
            #plt.show()
        st.write("A single image of all faces")    
        st.pyplot(fig)

        
    elif len(potraitfaces_list)==1:
        #fig, axes = plt.subplots(1, len(potraitfaces_list), figsize=(3, 3))
        axes.imshow(potraitfaces_list[0])
        axes.set_title(f'Potrait Image {1}')
        axes.axis('off')

        plt.tight_layout()
        #plt.show()
        #st.write("A single image of all faces")
        st.pyplot(fig,use_container_width=False)
        st.image(potraitfaces_list,use_column_width=False )
    else:
        st.write("No Potraits in image to display")


#config the webpage
st.set_page_config(page_title="Potrait Face APP", layout='wide')
st.header("Potrait Face")
st.subheader("Upload an image and get the potraits of all the faces in it!")

#streamlit file uploader

uploaded_file = st.file_uploader("Choose a file",type=['png', 'jpg'])

while uploaded_file:
    #read the image 
    img=Image.open(uploaded_file).convert('RGB')
    img_array=np.array(img)

    #call the class PotraitFace
    pf=PotraitFace(img_array)
    st.write("...starting to read the image")
    with  st.spinner("...getting the embeddings"):
        emb=pf.get_embeddings()
        if emb:
            st.write("...Got the embeddings")
        else:
            st.write("The image uploaded does not have a potrait. Upload another image")
            break
    with st.spinner("...getting the embeddings"):
        coordinates=pf.get_potraitface_coordinates()
    st.write("...Got the cordinates")
    
    with st.spinner("...getting the coordinates"):
        faces_list=pf.get_potraitfaces(coordinates)
    st.write(faces_list)
    st.write("...all potrait faces listed")
    with st.spinner("Displaying the Potraits"):
        if faces_list !=[]:
            plot_potraitfaces(faces_list)
        else:
            st.write("No Face could be detcted in the image uploaded.")
    st.write("Done!")
    break
else:
    st.write("upload the image file in jpg or png format")

end_time = time.time()
elapsed_time = end_time - start_time
st.write(f"Time taken to process: {elapsed_time} seconds")
  