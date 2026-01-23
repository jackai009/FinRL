FROM stablebaselines/rl-baselines3-zoo

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt -I

RUN pip install jupyterlab

# 添加Tini。Tini作为jupyter的进程子收割者运行。这可以防止内核崩溃。
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

# 复制代码而不是挂载
COPY . .

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''","--NotebookApp.password=''","--allow-root"]
