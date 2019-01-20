echo "--- Install it..."
./build.sh

echo "--- Run it..."
./build/CYK_CUDA $1 $2 $3
