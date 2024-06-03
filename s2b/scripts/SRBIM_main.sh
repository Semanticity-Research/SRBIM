
# CONFIG_FILE=s2b/config/config_3DBIT00_memorial.yaml
# CONFIG_FILE=s2b/config/config_3DBIT00_civil-public.yaml
# CONFIG_FILE=s2b/config/config_3DBIT00_commerical-office.yaml
# CONFIG_FILE=s2b/config/config_3DBIT00_residential.yaml
# CONFIG_FILE=s2b/config/config_Birmingham-block-0.yaml
# CONFIG_FILE=s2b/config/config_residential_church.yaml


PTH_PCD=data/s2b/3DBIT00-20240120/commerical-office-2.txt
TESTRUNs=1,3,5,7,9,11,1,3,5,7,9,11,1,3,5,7,9,11,1,3,5,7,9,11
PYTHON=python
# $PYTHON s2b/SRBIM_main.py \
# --config $CONFIG_FILE \
# --pth_pcd $PTH_PCD \
# --test_runs $TESTRUNs
python s2b/SRBIM_main.py --config $CONFIG_FILE 

echo "Running IFC Mesh Reconstruction with config file: $CONFIG_FILE"
