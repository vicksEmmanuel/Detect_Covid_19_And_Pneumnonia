import { Camera, CameraType } from 'expo-camera';
import React from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import useModel from './hooks/useModel';

import usePermissions from './hooks/usePermissions';

export default function App() {
  usePermissions();

  const {uploadAndProcessImage} = useModel();

  const processAndRunModel = async (image: Image) => {
    const value = await preprocessImage(image);
    console.log('value', value);

    const result = await getResult(value);

    console.log('result', result);

    if (result) {
      const classes = await understandResult(result);

      console.log('classes', classes);
    }
  };

  return (
    <View style={StyleSheet.absoluteFill}>
     <Camera style={[StyleSheet.absoluteFill]} type={CameraType.back}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={toggleCameraType}>
            <Text style={styles.text}>Flip Camera</Text>
          </TouchableOpacity>
        </View>
      </Camera>
    </View>
  );
}
