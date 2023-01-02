import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor
)
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.stages = {
            "Trans": config.Transformation,
            "Feat": config.FeatureExtraction,
            "Seq": config.SequenceModeling,
            "Pred": config.Prediction
        }

        """ Transformation """
        if config.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=config.num_fiducial,
                I_size=(config.imgH, config.imgW),
                I_r_size=(config.imgH, config.imgW),
                I_channel_num=config.input_channel
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if config.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(config.input_channel, config.output_channel)
        elif config.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(config.input_channel, config.output_channel)
        elif config.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(config.input_channel, config.output_channel)
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = config.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if config.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, config.hidden_size, config.hidden_size),
                BidirectionalLSTM(config.hidden_size, config.hidden_size, config.hidden_size))
            self.SequenceModeling_output = config.hidden_size
        else:
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if config.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, config.num_class)
        elif config.Prediction == "Attn":
            self.Prediction = Attention(self.SequenceModeling_output, config.hidden_size, config.num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages["Trans"] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages["Seq"] == "BiLSTM":
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.config.batch_max_length)

        return prediction
