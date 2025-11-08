---
title: C++通信原语
description: C++通信原语
author: cybotiger
date: 2025-10-30 12:00:00 +0800
categories: [计算机系统, 通信]
tags: []
math: true
mermaid: true
---

## std::mutex
最普通的锁，用来保护一个资源只有一个线程可以操作。

+ 缺点1：不安全，控制僵硬
    + 不能进行 recursive_lock（同个线程多次 lock 该锁）
    + 需要手动 lock 和 unlock；如果在持有锁的中途出现异常，可能会死锁；如果线程结束但仍然持有锁，会出现异常
    + 不支持 deferred lock，即先声明，后续通过 `lock()` 显式获取锁

+ 缺点2：效率低
    + 被其他线程占用时，`lock()` 会阻塞，该线程会进行 busy_waiting，导致 CPU 空转
    + 为了避免 CPU 资源的浪费，可以配合 `std::condition_variable` 使用

## std::unique_lock
mutex 的 RAII (Resource Acquisition Is Initialization) wrapper，支持 recursive_lock、deferred lock

## std::condition_variable
通过 wait 和 notify 机制，让等待锁的线程进入休眠状态，不占用计算资源，在 waiting_queue 中等待锁的释放

```cpp
/*
 * Wrapper class around a counter, a condition variable, and a mutex.
 */
class ThreadState {
    public:
        std::condition_variable* condition_variable_;
        std::mutex* mutex_;
        int counter_;
        int num_waiting_threads_;
        ThreadState(int num_waiting_threads) {
            condition_variable_ = new std::condition_variable();
            mutex_ = new std::mutex();
            counter_ = 0;
            num_waiting_threads_ = num_waiting_threads;
        }
        ~ThreadState() {
            delete condition_variable_;
            delete mutex_;
        }
};

void signal_fn(ThreadState* thread_state) {
    // Acquire mutex to make sure the shared counter is read in a
    // consistent state.
    thread_state->mutex_->lock();
    while (thread_state->counter_ < thread_state->num_waiting_threads_) {
        thread_state->mutex_->unlock();
        // Release the mutex before calling `notify_all()` to make sure
        // waiting threads have a chance to make progress.
        thread_state->condition_variable_->notify_all();
        // Re-acquire the mutex to read the shared counter again.
        thread_state->mutex_->lock();
    }
    thread_state->mutex_->unlock();
}

void wait_fn(ThreadState* thread_state) {
    // A lock must be held in order to wait on a condition variable.
    // This lock is atomically released before the thread goes to sleep
    // when `wait()` is called. The lock is atomically re-acquired when
    // the thread is woken up using `notify_all()`.
    std::unique_lock<std::mutex> lk(*thread_state->mutex_);
    thread_state->condition_variable_->wait(lk);
    // Increment the shared counter with the lock re-acquired to inform the
    // signaling thread that this waiting thread has successfully been
    // woken up.
    thread_state->counter_++;
    printf("Lock re-acquired after wait()...\n");
    lk.unlock();
}

/*
 * Signaling thread spins until each waiting thread increments a shared
 * counter after being woken up from the `wait()` method.
 */
void condition_variable_example() {
    int num_threads = 3;

    printf("==============================================================\n");
    printf("Starting %d threads for signal-and-waiting...\n", num_threads);
    std::thread* threads = new std::thread[num_threads];
    ThreadState* thread_state = new ThreadState(num_threads-1);
    threads[0] = std::thread(signal_fn, thread_state);
    for (int i = 1; i < num_threads; i++) {
        threads[i] = std::thread(wait_fn, thread_state);
    }
    for (int i = 0; i < num_threads; i++) {
        threads[i].join();
    }
    printf("==============================================================\n");

    delete thread_state;
    delete[] threads;
}
```

## std::atomic

如果需要保护的资源是一个简单的变量，可以使用更为简洁的 `std::atomic`，通常比 mutex 更快

```cpp
#include <iostream>
#include <thread>
#include <atomic>

std::atomic<int> counter(0);

void incrementCounter() {
    for (int i = 0; i < 1000; ++i) {
        ++counter;  // Atomic increment
    }
}

int main() {
    std::thread t1(incrementCounter);
    std::thread t2(incrementCounter);
    t1.join();
    t2.join();
    std::cout << "Final counter value: " << counter << std::endl;
    return 0;
}
```