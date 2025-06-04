export  PYTHONPATH=/project/video_training/customer_support/ndas/gradient-mechanics-wj/src/:$PYTHONPATH
echo $PYTHONPATH
pip install torchvision
pip install torchdata
pip install av
pip install cvcuda-cu12

python src/tests/benchmark_video_dataset.py  data/daytime-freeway.mp4   --num-workers  0
