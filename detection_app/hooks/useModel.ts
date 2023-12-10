/** @format */

import { useEffect, useState } from 'react';
import {
	Image,
	MobileModel,
	Tensor,
	media,
	torch,
	torchvision,
} from 'react-native-pytorch-core';

const T = torchvision.transforms;

const useModel = () => {
	const [ptModel, setPtModel] = useState<string | null>(null);
	const downloadModel = async () => {
		const url =
			'https://github.com/vicksEmmanuel/Detect_Covid_19_And_Pneumnonia/raw/main/model.pth';

		try {
			const filePath = await MobileModel.download(url);
			return filePath;
		} catch (error) {
			console.error('Error downloading file:', error);
		}
	};

	const initInferenceSession = async () => {
		try {
			const model = await downloadModel();

			setPtModel(model!);
		} catch (e) {
			console.log(e);
		}
	};

	useEffect(() => {
		initInferenceSession();
	}, []);

	return {
		ptModel,
		getResult: async (tensor: Tensor) => {
			try {
				const model = await torch.jit._loadForMobile(ptModel!);
				const output = await model.forward<any, Tensor>(tensor);
				return output;
			} catch (e) {
				console.log(e);
			}
		},
		preprocessImage: async (image: Image) => {
			const width = image.getWidth();
			const height = image.getHeight();

			const blob = media.toBlob(image);
			let tensor = torch.fromBlob(blob, [height, width, 3]);

			const resize = T.resize(224);
			const grayscale = T.grayscale(1);
			// tensor = tensor.permute([2, 0, 1]);
			// tensor = tensor.div(255);
			const centerCrop = T.centerCrop(Math.min(width, height));
			tensor = centerCrop(tensor);
			tensor = resize(tensor);
			tensor = grayscale(tensor);

			const normalize = T.normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			);
			tensor = normalize(tensor);
			tensor = tensor.unsqueeze(0);

			console.log(tensor, 'd=d-d=d');

			return tensor;
		},

		understandResult: (result: Tensor) => {
			const maxIdx = result.argmax().item();
			const classes = ['Covid', 'Normal', 'Pneumonia'];
			const className = classes[maxIdx];
			return className;
		},
	};
};

export default useModel;
