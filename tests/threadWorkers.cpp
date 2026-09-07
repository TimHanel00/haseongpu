#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <core/calcPhiAseThreaded.hpp>

#include <atomic>
#include <thread>
#include <vector>

TEST_CASE("thread worker cancellation releases waiting and late participants", "[forward][correctness][workers]")
{
    hase::core::detail::ThreadWorkerGroup group(3u);
    std::atomic<unsigned> failures = 0u;
    std::vector<std::jthread> workers;
    for(unsigned index = 0u; index < 3u; ++index)
        workers.emplace_back(
            [&, index]
            {
                try
                {
                    if(index == 1u)
                        throw std::runtime_error("injected worker failure");
                    (void) group.gather(index, index);
                }
                catch(std::runtime_error const&)
                {
                    ++failures;
                    group.cancel(std::current_exception());
                }
            });
    for(auto& worker : workers)
        worker.join();
    CHECK(failures == 3u);
}
