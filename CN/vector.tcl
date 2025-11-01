# Distance Vector Routing Simulation in NS2
# Bellman-Ford (RIP-style behavior)

# Create Simulator instance
set ns [new Simulator]

# Enable Distance Vector routing protocol
$ns rtproto DV

# Open NAM trace file
set nf [open dv.nam w]
$ns namtrace-all $nf

# Open Trace file
set tf [open dv.tr w]
$ns trace-all $tf

# Define finish procedure
proc finish {} {
    global ns nf tf
    $ns flush-trace
    close $nf
    close $tf
    exec nam dv.nam &
    exit 0
}

# Create 4 nodes
set n0 [$ns node]
set n1 [$ns node]
set n2 [$ns node]
set n3 [$ns node]

# Topology
# n0 -- n1 -- n2 -- n3
$ns duplex-link $n0 $n1 1Mb 10ms DropTail
$ns duplex-link $n1 $n2 1Mb 10ms DropTail
$ns duplex-link $n2 $n3 1Mb 10ms DropTail

# Orientation for visualization
$ns duplex-link-op $n0 $n1 orient right
$ns duplex-link-op $n1 $n2 orient right
$ns duplex-link-op $n2 $n3 orient right

# UDP Setup (n0 —(UDP/CBR)—> n3)
set udp0 [new Agent/UDP]
$ns attach-agent $n0 $udp0
set null0 [new Agent/Null]
$ns attach-agent $n3 $null0
$ns connect $udp0 $null0

# CBR Traffic Configuration
set cbr0 [new Application/Traffic/CBR]
$cbr0 attach-agent $udp0
$cbr0 set packetSize_ 500
$cbr0 set interval_ 0.2

# Optional: Link Failure & Recovery Events (to observe DV recalculation)
$ns rtmodel-at 1.5 down $n1 $n2
$ns rtmodel-at 2.5 up $n1 $n2

# Schedule Application Events
$ns at 0.5 "$cbr0 start"
$ns at 4.0 "$cbr0 stop"
$ns at 4.2 "finish"

# Run the simulation
$ns run
