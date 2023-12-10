import { Camera, CameraType } from 'expo-camera';
import React from 'react';
import { StyleSheet, TouchableOpacity, View } from 'react-native';
import useModel from './hooks/useModel';

import usePermissions from './hooks/usePermissions';

export default function App() {
  usePermissions();

  const {uploadAndProcessImage} = useModel();
  const cameraRef = React.useRef<Camera>(null);

  const captureAndProcess = async () => {
      if (cameraRef.current) {
        const image = await cameraRef.current.takePictureAsync({ base64: true });
        const result = await uploadAndProcessImage(image);

        console.log(result);
      }
  }

  return (
    <View style={StyleSheet.absoluteFill}>
     <Camera ref={cameraRef} style={[StyleSheet.absoluteFill]} type={CameraType.back}>
       
       <TouchableOpacity
       onPress={captureAndProcess}
        style={{position: 'absolute', bottom: 40, left: '43%', borderRadius: 100, borderWidth: 2, borderColor: 'white', padding: 2}}>
        <View style={{backgroundColor: 'white', height: 70, width: 70, borderRadius: 100,}}/>
       </TouchableOpacity>
      </Camera>
    </View>
  );
}
