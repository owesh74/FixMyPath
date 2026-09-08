package com.example.fixmypatha;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.location.Location;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationCallback;
import com.google.android.gms.location.LocationRequest;
import com.google.android.gms.location.LocationResult;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.location.Priority;

import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;
import java.io.InputStream;
import java.nio.FloatBuffer;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import okhttp3.ResponseBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * FixMyPath — MainActivity (CLEAN BUILD)
 *
 * All fixes applied:
 *  1. Single Retrofit instance, initialized once in onCreate.
 *  2. Single pothole-detection block in onSensorChanged — no duplicates.
 *  3. All variables properly declared (currentLat, currentLon, etc.).
 *  4. ONNX model with heuristic fallback.
 *  5. GPS via FusedLocationProviderClient with runtime permission.
 *  6. UI updates always on main thread.
 *  7. Proper sensor register/unregister lifecycle.
 *  8. ONNX resources closed in onDestroy.
 */
public class MainActivity extends AppCompatActivity implements SensorEventListener {

    private static final String TAG = "FixMyPath";
//    private static final String BASE_URL = "https://fixmypathserver.onrender.com/";
//    private static final String BASE_URL = "http://127.0.0.1:5000/";
private static final String BASE_URL = "http://192.168.46.251:5000/";
    private static final int LOCATION_PERMISSION_CODE = 101;

    // ── Retrofit ──────────────────────────────────────────────
    private ApiService apiService;

    // ── Sensors ───────────────────────────────────────────────
    private SensorManager sensorManager;
    private Sensor accelerometer;

    // ── UI ────────────────────────────────────────────────────
    private TextView statusText, dataText;
    private LinearLayout rootLayout;
    private final Handler uiHandler = new Handler(Looper.getMainLooper());

    // ── Timing ────────────────────────────────────────────────
    private long lastDetectionTime = 0;
    private long lastPredictionTime = 0;

    // ── GPS ───────────────────────────────────────────────────
    private FusedLocationProviderClient fusedLocationClient;
    private double currentLat = 0.0;
    private double currentLon = 0.0;

    // ── ONNX ──────────────────────────────────────────────────
    private OrtEnvironment ortEnv;
    private OrtSession ortSession;

    // ─────────────────────────────────────────────────────────
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // ── UI refs ───────────────────────────────────────────
        statusText = findViewById(R.id.statusText);
        dataText   = findViewById(R.id.dataText);
        rootLayout = findViewById(R.id.rootLayout);

        // ── Retrofit (single instance) ────────────────────────
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();
        apiService = retrofit.create(ApiService.class);

        // ── Accelerometer ─────────────────────────────────────
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        if (accelerometer == null) {
            Toast.makeText(this, "No accelerometer found!", Toast.LENGTH_LONG).show();
        }

        // ── GPS ───────────────────────────────────────────────
        fusedLocationClient = LocationServices.getFusedLocationProviderClient(this);
        requestLocationPermission();

        // ── ONNX model ────────────────────────────────────────
        loadOnnxModel();
    }

    // ── ONNX loader ───────────────────────────────────────────
    private void loadOnnxModel() {
        try {
            InputStream modelStream = getAssets().open("pothole_model.onnx");
            byte[] modelBytes = modelStream.readAllBytes();
            modelStream.close();

            ortEnv     = OrtEnvironment.getEnvironment();
            ortSession = ortEnv.createSession(modelBytes);
            Log.d(TAG, "✅ ONNX model loaded");
        } catch (Exception e) {
            Log.e(TAG, "❌ ONNX load failed — using heuristic fallback: " + e.getMessage());
            ortSession = null;
        }
    }

    // ── ML prediction (ONNX + heuristic fallback) ─────────────
    private double predict(float x, float y, float z, double magnitude) {
        if (ortSession != null) {
            try {
                OnnxTensor tensor = OnnxTensor.createTensor(
                        ortEnv,
                        FloatBuffer.wrap(new float[]{ x, y, z, (float) magnitude }),
                        new long[]{ 1, 4 });

                OrtSession.Result result = ortSession.run(
                        Collections.singletonMap("float_input", tensor));

                // Output[1] = probability array — index 1 = pothole class
                float[][] probs = (float[][]) result.get(1).getValue();
                float potholeProbability = probs[0][1];

                result.close();
                tensor.close();
                return potholeProbability;

            } catch (Exception e) {
                Log.w(TAG, "ONNX inference failed, using heuristic: " + e.getMessage());
            }
        }

        // Heuristic fallback
        double score = 0;
        if (magnitude > 10)          score += 0.3;
        if (Math.abs(x) > 2)         score += 0.2;
        if (Math.abs(y) > 2)         score += 0.2;
        if (Math.abs(z - 9.8f) > 2) score += 0.3;
        return Math.min(score, 1.0);
    }

    // ── GPS permission ────────────────────────────────────────
    private void requestLocationPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this,
                    new String[]{ Manifest.permission.ACCESS_FINE_LOCATION },
                    LOCATION_PERMISSION_CODE);
        } else {
            startLocationUpdates();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == LOCATION_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startLocationUpdates();
            } else {
                Toast.makeText(this,
                        "Location permission denied — coordinates will be 0, 0",
                        Toast.LENGTH_LONG).show();
            }
        }
    }

    private void startLocationUpdates() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION)
                != PackageManager.PERMISSION_GRANTED) return;

        LocationRequest request = new LocationRequest.Builder(
                Priority.PRIORITY_HIGH_ACCURACY, 3000)
                .setMinUpdateIntervalMillis(1000)
                .build();

        fusedLocationClient.requestLocationUpdates(request, new LocationCallback() {
            @Override
            public void onLocationResult(@NonNull LocationResult result) {
                Location loc = result.getLastLocation();
                if (loc != null) {
                    currentLat = loc.getLatitude();
                    currentLon = loc.getLongitude();
                    Log.d(TAG, "GPS updated: " + currentLat + ", " + currentLon);
                }
            }
        }, Looper.getMainLooper());
    }

    // ── Sensor lifecycle ──────────────────────────────────────
    @Override
    protected void onResume() {
        super.onResume();
        if (accelerometer != null) {
            sensorManager.registerListener(this, accelerometer,
                    SensorManager.SENSOR_DELAY_NORMAL);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (accelerometer != null) {
            sensorManager.unregisterListener(this);
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        try {
            if (ortSession != null) ortSession.close();
            if (ortEnv != null)     ortEnv.close();
        } catch (Exception e) {
            Log.e(TAG, "ONNX cleanup error", e);
        }
    }

    // ── Sensor events ─────────────────────────────────────────
    @Override
    public void onSensorChanged(SensorEvent event) {
        long currentTime = System.currentTimeMillis();

        // Throttle: run prediction at most every 300 ms
        if (currentTime - lastPredictionTime < 300) return;
        lastPredictionTime = currentTime;

        float x = event.values[0];
        float y = event.values[1];
        float z = event.values[2];
        double magnitude      = Math.sqrt(x * x + y * y + z * z);
        double potholeProb    = predict(x, y, z, magnitude);

        // Always update the sensor readout
        final String info = String.format(Locale.US,
                "X: %.2f  Y: %.2f  Z: %.2f\nMag: %.2f  Prob: %.2f",
                x, y, z, magnitude, potholeProb);
        uiHandler.post(() -> dataText.setText(info));

        // Cooldown: at most one detection event every 3 seconds
        if (currentTime - lastDetectionTime < 3000) return;

        // ── Detection threshold ───────────────────────────────
        if (potholeProb > 0.4 && magnitude > 11.5) {

            lastDetectionTime = currentTime;

            // Flash UI red
            uiHandler.post(() -> {
                statusText.setText("⚠️ Pothole Detected!");
                statusText.setTextColor(Color.WHITE);
                rootLayout.setBackgroundColor(Color.parseColor("#EF5350"));
            });

            // Build and send the report via Retrofit
            String timestamp = new SimpleDateFormat(
                    "yyyy-MM-dd HH:mm", Locale.getDefault()).format(new Date());

            PotholeReport report = new PotholeReport(
                    currentLat, currentLon, (float) magnitude, (float) potholeProb, timestamp);

            apiService.sendReport(report).enqueue(new Callback<ResponseBody>() {
                @Override
                public void onResponse(@NonNull Call<ResponseBody> call,
                                       @NonNull Response<ResponseBody> response) {
                    if (response.isSuccessful()) {
                        Log.d(TAG, "✅ Report sent to dashboard");
                    } else {
                        Log.w(TAG, "Server error: " + response.code());
                    }
                }

                @Override
                public void onFailure(@NonNull Call<ResponseBody> call,
                                      @NonNull Throwable t) {
                    Log.e(TAG, "❌ Network error: " + t.getMessage());
                }
            });

        } else {
            // Road is smooth
            uiHandler.post(() -> {
                statusText.setText("✅ Road Smooth");
                statusText.setTextColor(Color.BLACK);
                rootLayout.setBackgroundColor(Color.parseColor("#A5D6A7"));
            });
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) { }
}