set -e  # Exit immediately if a command fails

# -------------------------
# Environment configuration
# -------------------------
ENV_NAME="st_hcm_env"
PYTHON_VERSION="3.10"

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION" \
    numpy pandas scikit-learn matplotlib seaborn statsmodels tqdm \
    pytorch gpytorch \
    -c conda-forge -c pytorch

# -------------------------
# Activate and verify
# -------------------------
echo "Activating environment: $ENV_NAME"
source ~/miniforge3/bin/activate "$ENV_NAME"

# -------------------------
# Environment variable tuning
# -------------------------
echo "Appending recommended environment variables to ~/.zprofile (if not already set)"
{
    echo '# >>> Custom scientific computing settings >>>'
    echo 'export MKL_VERBOSE=NO'
    echo 'export OMP_NUM_THREADS=1'
    echo 'export KMP_DUPLICATE_LIB_OK=TRUE'
    echo '# <<< End custom settings <<<'
} >> ~/.zprofile

# -------------------------
# Completion message
# -------------------------
echo ""
echo "✅ Environment \"$ENV_NAME\" created successfully."
echo "👉 Please run: source ~/.zprofile"
echo "👉 Then activate your environment: conda activate $ENV_NAME"
echo ""
