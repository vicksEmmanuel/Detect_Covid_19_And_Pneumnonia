import { Camera, CameraType } from 'expo-camera';
import React from 'react';
import { ActivityIndicator, Alert, StyleSheet, TouchableOpacity, View } from 'react-native';
import useModel from './hooks/useModel';

import usePermissions from './hooks/usePermissions';

export interface Prediction {
  "result": {
    "class": "covid" | "normal" | "pneumonia", 
    "confidence": number
  }
}


export default function App() {
  usePermissions();

  const {uploadAndProcessImage} = useModel();
  const cameraRef = React.useRef<Camera>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const captureAndProcess = async () => {
      if (cameraRef.current) {
        const image = await cameraRef.current.takePictureAsync({ base64: true });
        setIsLoading(true);
        const result: Prediction = await uploadAndProcessImage(image) as any;
        setIsLoading(false);
        Alert.alert(
          "Results", 
          `This is a ${result?.result?.class} with ${result?.result?.confidence} confidence`
          );
      }
  }

  return (
    <View style={StyleSheet.absoluteFill}>
     <Camera ref={cameraRef} style={[StyleSheet.absoluteFill]} type={CameraType.back}>
       
       {isLoading &&  (
        <ActivityIndicator color={'red'}  size={50} style={{ top: '50%',position: 'absolute', right: '45%'}}/>
       )}
       <TouchableOpacity
       onPress={captureAndProcess}
       disabled={isLoading}
        style={{position: 'absolute', bottom: 40, left: '43%', borderRadius: 100, borderWidth: 2, borderColor: 'white', padding: 2}}>
        <View style={{backgroundColor: 'white', height: 70, width: 70, borderRadius: 100,}}/>
       </TouchableOpacity>
      </Camera>
    </View>
  );
}
