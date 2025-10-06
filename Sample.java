class Sample {
    Sample() {}
    int add(int a, int b) { return a + b; }
    int minus(int a, int b) { return a - b; }
    static { System.out.println("init"); }  // initializer (no name)
}


