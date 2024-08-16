#include "Thread_Pool.h"

Thread_Pool::Thread_Pool() : stop(true) {}

void Thread_Pool::setupPool(size_t threads) {
    stop = false;
    for (size_t i = 0; i < threads; ++i) {
        workers.emplace_back(&Thread_Pool::worker, this);
    }
}

void Thread_Pool::clearPool() {
    
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true; 
        condition.notify_all(); 
    }

    for (std::thread &worker : workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();

    std::queue<std::function<void()>> empty;
    std::swap(tasks, empty); 

    activeTasks = 0;

    finishedCondition.notify_all();    
}

Thread_Pool::~Thread_Pool() {
  
    clearPool();
}

void Thread_Pool::wait() {

    std::unique_lock<std::mutex> lock(queueMutex);

    finishedCondition.wait(lock, [this] { 
        return stop || (tasks.empty() && activeTasks == 0); 
    });
}

void Thread_Pool::worker() {

     while (true) {

        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            condition.wait(lock, [this]() { 
                return stop || !tasks.empty(); 
            });

            if (stop) {
                return; 
            }

            task = std::move(tasks.front());
            tasks.pop();
        }

        task();
        
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            activeTasks--;
            finishedCondition.notify_one(); 
        }
    }
}