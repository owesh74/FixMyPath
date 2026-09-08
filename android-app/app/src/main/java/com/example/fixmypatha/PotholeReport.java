package com.example.fixmypatha;

public class PotholeReport {
    public double latitude;
    public double longitude;
    public double magnitude;
    public double probability;
    public String timestamp;

    public PotholeReport(double lat, double lon, double mag, double prob, String time) {
        this.latitude    = lat;
        this.longitude   = lon;
        this.magnitude   = mag;
        this.probability = prob;
        this.timestamp   = time;
    }
}
