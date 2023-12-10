/** @format */

import * as Permissions from 'expo-permissions';
import { useEffect } from 'react';
import { Linking } from 'react-native';

const usePermissions = () => {
	const requestPermissions = async () => {
		const { status } = await Permissions.askAsync(Permissions.CAMERA);

		if (status !== 'granted') {
			Linking.openSettings();
		}
	};

	useEffect(() => {
		requestPermissions();
	}, []);

	return {};
};

export default usePermissions;
