# Development Guide

## Function Chain


- run_scheduler_process()
    - scheduler = Scheduler(...)
    - scheduler.event_loop_normal()
        - recv_reqs = recv_requests()
        - process_input_requests(recv_reqs)
        - batch = get_next_batch_to_run() -> ScheduleBatch
        - 



-> scheduler.process_input_requests()
-> scheduler.get_next_batch_to_run()
-> scheduler.run_batch()
-> scheduler.tp_worker.forward_batch_generation()
-> tp_worker.model_runner.forward()
-> model_runner._forward_raw()
-> model_runner.forward_extend()/forward_decode()
-> model_runner.model.forward()