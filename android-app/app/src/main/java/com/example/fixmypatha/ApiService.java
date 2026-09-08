package com.example.fixmypatha; // Make sure this matches your package name

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.http.Body;
import retrofit2.http.POST;

public interface ApiService {

    @POST("report")
    Call<ResponseBody> sendReport(@Body PotholeReport report);
}