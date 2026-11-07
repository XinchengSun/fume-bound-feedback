# GPU1 的实际启动入口

以下命令由执行端启动脚本和服务器保留的文件还原。训练工作目录位于同一个共享盘。首次和第二次均使用同一作业 worker-0、物理 GPU1，其 UUID 见 acceptance150.json。服务器归档的 worker 合并输出见 compare150-worker.log；它同时记录 B 首次初始化验证时的异常。平台认证与终端通道位于启动器外部。

~~~sh
own=/inspire/ssd/project/video-generation/public/sunxincheng
r=$own/workspaces/635fe852-6e5f-48e2-b5b1-72b10d58561b/requests/sxc-4090x2-635fe852-r2
v=$own/FUME/local_ops/v5_20261106/validation

# 首次；完成 A，首次 B 在验证时退出
nohup bash "$r/with-one-gpu.sh" 1 sh "$r/run-python.sh" -u -B "$v/compare150.py" >>"$v/compare150-worker.log" 2>&1 </dev/null &

# 恢复；完成 B、C、D
sh "$v/resume_gpu1.sh"
~~~

第二条命令对应本目录 resume_gpu1.sh 的原件。脚本导出 FUME 独立 NumPy 2.2.6 的 PYTHONPATH，并在同一 GPU1 包装器下执行 recover150.py。入口所做的 GPU 选择和解释器装载见 with-one-gpu.sh 和 run-python.sh。初次失败的 provenance.json 及 traceback 留在 provenance/first_attempt_initial_validation/。B/C/D 的实际训练内层调用 runner.main(args)，精确参数构造和测试 checkpoint 路径见 recover150.py 的 run_v5。
