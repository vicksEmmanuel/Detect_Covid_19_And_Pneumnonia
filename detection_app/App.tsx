import { Camera, CameraType } from 'expo-camera';
import React, { useEffect } from 'react';
import { StyleSheet, View } from 'react-native';
import useModel from './hooks/useModel';

import usePermissions from './hooks/usePermissions';

export default function App() {
  usePermissions();

  const {uploadAndProcessImage} = useModel();
  const cameraRef = React.useRef<Camera>(null);

  useEffect(() =>{
    const interval = setInterval(async () => {
      if (cameraRef.current) {
        const image = await cameraRef.current.takePictureAsync({ base64: true });
        const result = await uploadAndProcessImage(image);
      }
    }, 5000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <View style={StyleSheet.absoluteFill}>
     <Camera ref={cameraRef} style={[StyleSheet.absoluteFill]} type={CameraType.back}>
       
      </Camera>
    </View>
  );
}
