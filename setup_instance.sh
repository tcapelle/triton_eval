# apt install
apt install -y tmux curl git


# install python.h
apt-get install python3.10-dev

# git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs

# setup git
git config --global user.email "tcapelle@pm.me"
git config --global user.name "Thomas Capelle"

# Install gh
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
	&& mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y

# Instal UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# add venv management to bashrc:
echo "
# Load virtual environment - try local .venv first, fallback to base
if [ -f ./.venv/bin/activate ]; then
    source ./.venv/bin/activate
elif [ -f ~/base/bin/activate ]; then
    source ~/base/bin/activate
fi
" >> ~/.bashrc

echo "export VLLM_USE_V1=0" >> ~/.bashrc
# TRL setup
# pip install git+https://github.com/huggingface/trl.git@0dad4eb7ca8de6f93a76752a5773c0baecd4a3d3