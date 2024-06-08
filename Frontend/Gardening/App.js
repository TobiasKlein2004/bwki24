import { Camera, CameraView, useCameraPermissions, takePictureAsync } from 'expo-camera';
import { Text, View, Button, TouchableOpacity, Image, ImageBackground } from 'react-native';
import { manipulateAsync, ActionCrop, SaveFormat } from 'expo-image-manipulator';
import { useState, useRef } from 'react';
import * as FileSystem from 'expo-file-system'

const { styles } = require('./Styles.js')

export default function App() {
  const [permission, requestPermission] = useCameraPermissions();

  const [previewVisible, setPreviewVisible] = useState(false)
  const [capturedImage, setCapturedImage] = useState(null)

  const cameraRef = useRef(null);

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center' }}>We need your permission to show the camera</Text>
        <Button onPress={requestPermission} title="grant permission" />
      </View>
    );
  }




  const __sendImageToServer = async (image) => {
    console.log(image)

    const fileInfo = await FileSystem.getInfoAsync(image);

    if (!fileInfo.exists) {
      console.error('File does not exist at the specified path');
      return;
    }

    const formData = new FormData();
    formData.append('image', {
      uri: image,
      type: 'image/jpeg',
      name: 'image_' + new Date().getTime() + '.jpg',
    });

    const response = await fetch('http://192.168.1.66:5000/upload', {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
    });

    const responseJson = await response.json();
    console.log(responseJson);
  }



  const __cropPicture = async (image) => {
    console.log('crop');
    let width = 0;
    let height = 0;
    
    // Await Image.getSize using a Promise
    await new Promise((resolve, reject) => {
      Image.getSize(image.uri, (w, h) => {
        width = w;
        height = h;
        resolve();
      }, (error) => {
        console.error('Failed to get image size:', error);
        reject(error);
      });
    });
  
    const manipResult = await manipulateAsync(
      image.localUri || image.uri,
      [{
        crop: {
          height: width,
          originX: 0,
          originY: Math.floor(height * 0.2),
          width: width,
        }
      }],
      { compress: 1, format: SaveFormat.JPG }
    );
  
    return manipResult;
  };



  const __takePicture = async () => {
    const photo = await cameraRef.current.takePictureAsync()
    setPreviewVisible(true)
    setCapturedImage(photo)
    // Await __cropPicture before passing the result to __sendImageToServer
    const croppedImage = await __cropPicture(photo);
    await __sendImageToServer(croppedImage.uri);
  }



  const __retakePicture = () => {
    setCapturedImage(null)
    setPreviewVisible(false)
  }


  
  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef} facing={'back'}/>

      {
        previewVisible && capturedImage ? 
        (<CameraPreview photo={capturedImage}/>) :
        (<></>)
      }

      <View style={styles.blackOverlayBottom}></View>
      <View style={styles.blackOverlayTop}></View>

      {
        previewVisible && capturedImage ? 
        (
          <TouchableOpacity 
            style={styles.button} 
            onPress={__retakePicture}>
            <Image
              style={styles.cameraImage}
              source={require('./assets/redo.png')}
            />
          </TouchableOpacity>
        ) :
        (
          <TouchableOpacity 
            style={styles.button} 
            onPress={__takePicture}>
            <Image
              style={styles.cameraImage}
              source={require('./assets/camera.png')}
            />
          </TouchableOpacity>
        )
      }

    </View>
  );
}



const CameraPreview = ({ photo }) => {
  return (
    <View style={styles.preview}>
      <ImageBackground source={{uri: photo && photo.uri}} style={styles.previewImage} />
    </View>
  );
};
