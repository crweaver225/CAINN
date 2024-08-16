#ifndef THREAD_POOL_H_
#define THREAD_POOL_H_

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <functional>
#include <condition_variable>
#include <atomic>
#include <future>
#include <iostream>

class Thread_Pool {
public:
    Thread_Pool();
    ~Thread_Pool();

    void setupPool(size_t threads);
    void clearPool();

    template<typename F, typename... Args>
    void enqueue(F&& f, Args&&... args) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);
            tasks.emplace([f = std::forward<F>(f), ...args = std::forward<Args>(args)]() mutable {
                if constexpr (std::is_member_function_pointer<F>::value) {
                    std::invoke(f, std::forward<Args>(args)...);
                } else {
                    f(std::forward<Args>(args)...);
                }
            });
            activeTasks++;
        }
        condition.notify_one(); 
    }

    void wait();

private:
    std::vector<std::thread> workers; 
    std::queue<std::function<void()>> tasks; 

    void worker();

    std::mutex queueMutex;
    std::condition_variable condition; 
    std::condition_variable finishedCondition; 
    std::atomic<bool> stop{false}; 
    std::atomic<int> activeTasks{0}; 
};

#endif