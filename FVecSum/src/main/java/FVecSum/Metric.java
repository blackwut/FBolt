package FVecSum;

import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;

// Metric class
public class Metric implements Serializable {
    private String name;
    private String fileName;
    private ArrayList<Double> descriptiveStatistics;
    private long total;

    private Double mean;
    private Double mean_last_2000;
    private Double min;
    private Double max;

    // constructor
    public Metric(String name) {
        this.name = name;
        fileName = String.format("metric_%s.json", name);
        this.descriptiveStatistics = new ArrayList<Double>();
        this.mean = Double.MAX_VALUE;
        this.min = Double.MAX_VALUE;
        this.max = Double.MIN_VALUE;
        this.mean_last_2000 = Double.MAX_VALUE;
    }

    // add method
    public void add(double value) {
        descriptiveStatistics.add(value);
    }

    // setTotal method
    public void setTotal(long total) {
        this.total = total;
    }

    private void update() {
        Double sum = 0.0;
        Double sum_last = 0.0;
        int size = descriptiveStatistics.size();
        for (int i = 0; i < size; i++) {
            Double d = descriptiveStatistics.get(i);
            sum += d;
            if (this.min > d) this.min = d;
            if (this.max < d) this.max = d;
            if (i >= (size - 2000 + 1)) {
                sum_last += d;
            }
        }
        this.mean =  sum / size;

        if (size >= 2000) {
            this.mean_last_2000 = sum_last / 2000;
        } else {
            this.mean_last_2000 = sum_last / size;
        }
    }

    // dump method
    public void dump() throws IOException {
        this.update();
        FileWriter writer = new FileWriter(fileName);
        writer.write("name: " + name + ", \n");
        writer.write("samples: " + descriptiveStatistics.size() + ", \n");
        writer.write("total: " + total + ", \n");
        writer.write("mean: " + this.mean + ", \n");
        writer.write("mean(last 2000): " + this.mean_last_2000 + ", \n");
        writer.write("min: " + this.min + ", \n");
        writer.write("max: " + this.max + ", \n");
        writer.close();
    }
}
