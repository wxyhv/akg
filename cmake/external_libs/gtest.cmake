set(gtest_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")
set(gtest_CFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all -Wl,-z,relro,-z,now,-z,noexecstack")

if(ENABLE_GITEE)
    set(GTEST_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.8.0.tar.gz")
    set(GTEST_MD5 "89e13ca1aa48d370719d58010b83f62c")
else()
    set(GTEST_URL "https://github.com/google/googletest/archive/release-1.8.0.tar.gz")
    set(GTEST_MD5 "16877098823401d1bf2ed7891d7dce36")
endif()

akg_add_pkg(gtest
        VER 1.8.0
        LIBS gtest
        URL ${GTEST_URL}
        MD5 ${GTEST_MD5}
	CMAKE_OPTION -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF
	-DCMAKE_MACOSX_RPATH=TRUE -Dgtest_disable_pthreads=ON)
include_directories(${gtest_INC})
add_library(akg::gtest ALIAS gtest::gtest)

file(COPY ${gtest_LIBPATH}/libgtest.a DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
file(COPY ${gtest_LIBPATH}/libgtest_main.a DESTINATION ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest)
