from Preprocessor import Preprocessor
from train.AETrainer import AETrainer
from datasets.STL10 import STL10
from models.AutoEncoder import AutoEncoder

target_shape = [96, 96, 3]
model = AutoEncoder(num_layers=4, batch_size=128, target_shape=target_shape, tag='default')
data = STL10()
preprocessor = Preprocessor(target_shape=target_shape, augment_color=True)
trainer = AETrainer(model=model, dataset=data, pre_processor=preprocessor, num_epochs=500, lr_policy='linear',
                    optimizer='adam', init_lr=0.0003, num_gpus=2)
trainer.train_model(None)
