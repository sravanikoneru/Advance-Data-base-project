python train.py --dataset=cora --dropout=0.5 --weight_decay=5e-3 --epochs=500
python train.py --dataset=enzymes --weight_decay=5e-3 --num_layers=3 --epochs=500


python train.py --model_type GraphSage --hidden_dim 256 --dropout 0.5 --weight_decay 5e-3 --epochs=500 
python train.py --model_type GraphSage --hidden_dim 256 –dataset enzymes --dropout 0 --weight_decay=5e-3 --num_layers=3 --epochs 500 

python train.py --model_type GAT --lr 0.001 --hidden_dim 64 --dropout 0.5 --weight_decay 5e-3 --epochs=500  
python train.py --model_type GAT --dataset enzymes --dropout 0 --weight_decay=5e-3 --num_layers=3 --epochs 500  

