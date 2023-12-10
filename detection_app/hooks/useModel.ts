/** @format */

import { Platform } from 'react-native';

const useModel = () => {
	const uploadAndProcessImage = (file: any) => {
		return new Promise((resolve, reject) => {
			let url = `https://illness-detector-1gyp.onrender.com/predict`;

			const fileData: any = new FormData();

			let finalUri = file.uri;

			if (Platform.OS === 'android' && !file.uri.startsWith('file://')) {
				finalUri = 'file://' + file.uri;
			}

			if (Platform.OS === 'ios' && !file.uri.startsWith('file:///')) {
				finalUri = 'file://' + file.uri;
			}

			fileData.append('file', {
				uri: finalUri,
				type: 'image/jpeg',
				name: finalUri?.substring(finalUri.lastIndexOf('/') + 1),
			});

			fetch(url, {
				method: 'POST',
				headers: {
					accept: '*/*',
					'Content-Type': 'multipart/form-data',
				},
				body: fileData,
			})
				.then((response) => {
					if (!response.ok) {
						reject(response);
					}
					return response.json();
				})
				.then((data) => {
					return resolve(data?.imageUrl);
				})
				.catch((error) => {
					console.error(
						'There was a problem with the fetch operation:',
						JSON.stringify(error)
					);
					reject(error);
				});
		});
	};

	return {
		uploadAndProcessImage,
	};
};

export default useModel;
