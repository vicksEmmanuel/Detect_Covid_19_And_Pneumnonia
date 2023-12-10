import React from 'react';
import { StyleSheet, View } from 'react-native';
import { Camera, Image } from 'react-native-pytorch-core';
import useModel from './hooks/useModel';

import usePermissions from './hooks/usePermissions';

export default function App() {
  usePermissions();

  const {preprocessImage, getResult, understandResult} = useModel();

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
      <Camera
        style={[StyleSheet.absoluteFill, {backgroundColor: 'black'}]}
        onCapture={processAndRunModel}
      />
    </View>
  );
}
