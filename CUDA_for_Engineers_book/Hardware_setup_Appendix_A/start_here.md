
A cuda-enabled parallel computing system includes both hardware and software components,and here we deal with the necessary hardware component ----> " a CUDA-enabled GPU " .

# GPU Computing Ecosystems

This document highlights the differences between the primary high-performance GPU programming stacks: NVIDIA's CUDA, AMD's ROCm, and Intel's oneAPI.

## Comparison Summary

| Feature | NVIDIA | AMD | Intel |
| :--- | :--- | :--- | :--- |
| **Primary Stack** | CUDA | ROCm | oneAPI |
| **Programing Language** | CUDA C++ | HIP / C++ | SYCL / C++ |
| **Portability** | Fixed (NVIDIA only) | High (Can run on NVIDIA/AMD) | Universal (CPU, GPU, FPGA) |
| **Library Maturity** | Industry-leading | Very Strong (AI-focused) | Growing (Science-focused) |
| **Best For** | Max performance, AI Training | Cost-effective AI, Open source | Cross-platform, Enterprise |

## Deep Dive

### NVIDIA: CUDA (Proprietary)
CUDA is the industry standard for high-performance computing, but it is locked to NVIDIA hardware. It offers the most mature ecosystem with highly optimized libraries like cuBLAS and cuDNN.

### AMD: ROCm & HIP (Open/Portable)
AMD uses ROCm as its primary stack. The **HIP (Heterogeneous-computing Interface for Portability)** C++ runtime allows developers to write code that can run on both AMD and NVIDIA GPUs. The `HIPIFY` tool can automatically port most CUDA codebases to HIP.

### Intel: oneAPI & SYCL (Universal)
Intel's oneAPI aims for cross-architecture unity (CPUs, GPUs, and FPGAs). It uses **SYCL**, a standard-based C++ programming model that is fully hardware-agnostic.

---

1.First,we should start on how to determine if our system has a cuda-enabled GPU or not ,and this procedure depends on operating system (windows,OS X , Linux) that applies to your system ; 

### Checking for CUDA-enabled/NVIDIA GPU on Windows : 
---> Right-click on the desktop,if th epop-up menu doesnt have an entry for the " NVIDIA Control Panel "  ,then read "Upgrading Compute Capability" topic below ,and if  " NVIDIA Control Panel " is available , click to open it and then click the " Home " icon  , and a sample "Home window " is shown , the bottle line of the green " NVIDIA CONTROL  Panel" rectangle shows the model of the NVIDIA GPU installed on the system(ex : Geforce RTX 3060) and once identified the GPU model , proceed to the topic "Determining Compute Capability" below . 

### Checking for CUDA-enabled/NVIDIA GPU on OS X : 
---> From the Apple menu,select "About This Mac" where the "Displays" tab provides information about both the monitor and GPU on your system , note the model of your graphics  card (gpu-graphics processing unit) (if you  have one) and proceed to the topic  "Determining Compute Capability" below , and if no NVIDIA GPU is listed , proceed to the section on "Upgrading Compute Capability" below . 

### Checking for CUDA-enabled/NVIDIA GPU on Linux : 
From the command line [which can be accessed under Ubuntu via the keyboard shortcut Ctrl+Alt+t],enter the following command : 

" lspci | grep -i nvidia " 

---> to produce a list of peripheral  devices installed on your system ("ls" is short for "list" and "pci" is the "communications-bus" that connects between the CPU and peripheral devices such as your graphics card(GPU) and  "-i" flag is given to "grep" to ignore the case-sensitivity of the search term "nvidia" , and "grep" is a command that searches for a specific pattern in a file or stream of text and the full list of installed PCI devices  is piped to "grep" ---> the pattern-matching tool,to search the list for nvidia in a case-sensitive manner,as indicated by " -i " ) 

-on my system it produced the following output : 

" 01:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate] (rev a1) " ---> indicating the presence of a GeForce RTX 3060  graphics card  ,
and if your system has no installed NVIDIA card,proceed to the topic "Upgrading-Compute-Capability" below or else if you do have installed NVIDIA card ,note the model and proceed to the section -
-on "Determining-Compute-Capability" below .


### Determining Compute Capability of  NVIDIA-GPU on Linux : 
---> to find the compute capability of an NVIDIA-GPU on Linux , you can use several tools epending on whether you have the NVIDIA driver or the full CUDA Toolkit installed : 

1. Using nvidia-smi (If you have the NVIDIA driver installed (even without the full CUDA Toolkit), use this command and best for drivers only) : 

Command : " nvidia-smi --query-gpu=compute_cap --format=csv  "
 and if you want name too of that gpu/graphics-card ---> you can try this command :
 command : "nvidia-smi --query-gpu=name,compute_cap --format=csv " 
 --->outputs the GPU name and its compute capability (and this specific query field requires CUDA Toolkit 11.6+ or modern-drivers ,check  "nvidia-smi --help" for more details )
  and without that " --format=csv " flag too, you can try ,it will give the same result , and  if any doubts on these commans ,use this command : " nvidia-smi --help " 

2. Using "deviceQuery" (Best for CUDA Toolkit) : 
--->If you have the CUDA Toolkit installed , NVIDIA provides a utility specially for detailed hardware information ,
--->Location : Usually found at " /usr/local/cuda/extras/demo_suite/deviceQuery " 
try this command : " /usr/local/cuda/extras/demo_suite/deviceQuery | grep "Capability" "if its cuda-12.8 --->mention cuda-12.8 instead of just "cuda" and if its cuda-13.0 that u r having,mention cuda-13.0 insead of just "cuda" in above command ,and so on ...,
--->this directly shows the  "CUDA Capability Major/Minor version number " (ex: my terinal showed as "8.6" in which 8 is the major version number and 6 is the minor version number ,and both of these version numbers tells us the Architecture Generation and incremental improvements respectively (search in nvidia documentaions for more info )) ,

3. Using "nvcc" (To check supported architectures) : 
---> To see which compute capabilities your installed compiler can target for building software : 
use this command : " nvcc --help | grep -A 20 "gpu-architecture" " 
and if you want to  make sure your terminal always points to the correct and newest CUDA version without typing long paths , run this command to see where your system thinks nvcc is living : 
command : "which nvcc"
(ex : my terminal gave this : " /usr/local/cuda-13.0/bin/nvcc " --->indicating my system is using cuda-13.0) and 

### Quick Reference : Compute Capabilities of NVIDIA GPU's : 

| GPU Architecture | Compute Capability | GPU Models |
| :--- | :--- | :--- |
| Tesla | 1.x | Tesla C1060 |
| Fermi | 2.x | GTX 500 series |
| Kepler | 3.x | GTX 700 series (some models) |
| Maxwell | 5.x | GTX 900 series |
| Pascal | 6.x | GTX 10 series, Titan X (Pascal) |
| Volta | 7.0 | Tesla V100 (The first "Tensor Cores GPU") |
| Turing | 7.5 | RTX 20 series, Titan RTX, T4 (RTX was born here) |
| Ampere | 8.0, 8.6 | RTX 30 series, A100 |
| Ada Lovelace | 8.9 | RTX 40 series/L-40S |
| Hopper | 9.0 | H100, H200 |
| Blackwell | 10.0 | B100, B200 |

if the OS is other than linux then : visit "https://developer.nvidia.com/cuda-gpus"  and then go to the appropriate column  to find the compute capability of your GPU ;  

### Hardware Nomenclature : 
--->Names like Geforce RTX 3060 actually designate a graphics card ,   not gpu itself ,in my case the graphics card includes a model GA106 gpu that has "Ampere class architecture " and compute-capability 8.6 , but when discussing CUDA  hardware , there are 4-major designations : 
1.Model name (ex : Geforce RTX 3060 in my case) ; 
2.Model number (ex : GA106 chip in my case) ; 
3.Compute capability (ex : 8.6 in my case) ;  
4.Architecture class (ex : Ampere class architecture in my case) .  

now you may ask a doubt like whats this model number?Is that chip is what we mean by actual gpu or what?Can i say i have a GA106 chip or GA106GPU?or else both are different or same?or else all thjose 4 components/designations mentioned above are combinedly a GPU?or else Geforce RTX 3060(in my case) is a what the real GPU?if no t,whats this GeForce RTX 3060 called?
---> see for all the questions asked above, we shouldnt get confused and mix things up ,for clear clarity we should move ourselves from "Consumer language"(what gamers say/call) to "Engineer Language"(what CUDA developers say) :
# 1.The GeForce RTX 3060 is the Graphics card : 
In professional circles , this is the Card or the Board (or sometimes just "the GPU" in casual conversation) , 
---> what it is : The physical product you hold in your hand, and it includes GA106 chip,the 12GB of VRAM chips,the fans,the plastic shroud,the RGB lights, and the HDMI ports , 
---> Analogy : Think of it like a "Computer" (the whole box with fans, lights, and ports) vs the "CPU" (the chip inside)  or else this is a car (the whole body with wheels,seats,lights,and ports) vs the engine (the chip inside) , 
---> So, when you say "I have a GeForce RTX 3060", you are referring to the entire graphics card (the whole package) , not just the chip inside it . 

# 2.The GA106 is the GPU (The chip/gpu chip) : 
when the author says "GPU" ,he is specifically talking about the silicon chip soldered onto the center of that board ,
---> what it is : This is the "brain ", and its the actual square piece of silicon where the math happens and with this chip in my system, now I can absolutely say,"I have a GA106 GPU ",thats 100% technically correct, but its better to say "I have GA106 chip" or "I have GA106 GPU chip" to avoid confusion , 
---> So, when someone says "I have a GA106 GPU", he/she is referring to the actual silicon chip (the brain) , not the entire graphics card (the whole package) , 
---> Analogy : Think of it like a "CPU" (the chip inside) vs a "Computer" (the whole box with fans, lights, and ports/Graphics card) or else this is an engine (the chip inside) vs a car (the whole body with wheels,seats,lights,and ports/Graphics card) . 

# 3.Ampere is the Architecture : 
This is not a physical thing you can touch; it is the blueprint design , 

  ---> What it is: It describes how the transistors inside the GA106 are organized. Every GA106 chip is built using the Ampere blueprint , 
  ---> Analogy: This is the Technology Platform (e.g., "Internal Combustion Engine technology") . 

# 4.Compute Capability is the Feature Set : 
Think of this as the Software Version of the Hardware.

   ---> What it is : It tells the compiler (nvcc) what "tricks" the GA106 can do. It’s like a list of supported commands.
   ---> Analogy : This is the Fuel Requirement or the Operating Manual (e.g., "Requires 93 Octane fuel").

# 5.What does the author mean by GPU? : 
When the author says, "Names like GeForce RTX 3060 actually designate a graphics card, not the GPU itself," he means: 

The GPU is the GA106 , and he wants us to stop thinking about the "Box Name" and start thinking about the "Silicon Name"
Why? Because of "The Mix-up" : 
NVIDIA sometimes puts different GPUs inside the same Card Name : 

   ---> Some RTX 3060 Cards were made with a GA106 GPU , and other RTX 3060 Cards were made with a "cut down" GA104 GPU (a bigger chip that was slightly broken, so they disabled parts of it to make it act like a 3060),
and if we are the engineers writing a low-level driver, we shouldn't care what  that the box says "3060" or some other thing ,we just  need to know if we are talking to a GA106 or a GA104 , 
so,
He is warning us that one name doesn't tell the whole story, 
For example :  there are two versions of the RTX 3060 , one has a GA106 chip and one has a GA104 chip,and they are both called "RTX 3060" on the box, but their "internal designation" is different,
---> whas the difference btn gpu's of different architectures : 
(ex : Geforce RTX 3060's GA106 gpu vs ADA RTX 6000's AD102 gpu) : 
    Ampere Architecture gpu's ranges in  compute capabilities from 8.0 to 8.6  and -
   - Ada Lovelace Architecture (like the RTX 6000 Ada) is CC 8.9 , 

NVIDIA chose the version number 8.9 for "Ada" because it is a massive evolution of Ampere (8.x), but not a "ground-up" total rewrite like 9.0 (Hopper) , and t understand the differences btn two diff architectures gpu's ,lets first understand the differnces btn two same architectures gpu's :  
The Difference in "DNA" : 
If two cards are both 8.6 (here  are the specific GA10x series chips used in each card: 

    RTX 3090 / 3090 Ti : Uses the GA102 chip (specifically GA102 ) ; 
    RTX 3080 / 3080 Ti : Uses the GA102 chip (specifically GA102 ) ; 
    RTX 3070 / 3070 Ti : Uses the GA104 chip ; 
    RTX 3060 / 3060 Ti : Uses the GA106 or GA104 (for Ti variant) chips ) ; 

    The Layout (DNA) is IDENTICAL :  all of the  above chips (GA102,GA104,GA106) have the exact same "blueprint" for a single Streaming Multiprocessor (SM) , those all  handle math the same way , but 
    The Quantity is DIFFERENT :  Think of it like a Hotel ,  the 3060 is a hotel with 28 rooms (SMs) and  the 3080 is the same hotel design , but it has 68 rooms (SMs) , 
    The Structure : The "hallways" (data paths) and "utilities" (cache) are the same , there are just more of them in the higher-end card , 
    Small chip(GA106(my RTX 3060 chip)) vs Large chip(GA102(RTX 3090 chip)) : 
    for suppose ,lets say if my --->GA106 : 28 SM's , 192-bit (narrower "road" for data),~170 Watts power-draw  ,then Large-chip(GA102,RTX-3090 GPU)--->82 SM's,384-bit(wider "road" for data),~350 Watts power-draw ---> though the compute capability will be same,the performance will be different becoz of these different quantities of SM's and memory bus width and power draw etc ., and then , 
    both of these are CC 8.6 , this means they speak the exact same dialect of the Ampere language : 
    the 3090 just has more "mouths" (cores) speaking it at the same time  whereas the 3060 has fewer "mouths"(cores) but the words they say are identical ,  
    ---> Why do they do this : 
    It's about manufacturing yield ,it is very hard to bake a "perfect" giant chip without a single dust particle ruining it and it's much easier to bake smaller chips ,so, they design a "Family" of chips (GA102, 104, 106) that all speak the same language (CC 8.6) but have different "muscle" levels , 

and now you may ask  doubt  What changes when the "Architecture" changes (8.6 --->8.9)?
When we move from Ampere (8.6) to Ada (8.9), the actual "DNA" changes : 

    New Features: Ada added "Shader Execution Reordering" (SER) ,  Ampere physically cannot do this and ,it’s like adding a new brain function ; 
    Instruction Efficiency: A single core in an 8.9 chip might be 2x faster at a specific type of AI math than a core in an 8.6 chip, even if they are clocked at the same speed ; 
    The Version Number: That’s why the book focuses on CC , and  if you write code using an "89-only" feature, your 8.6 card will say : "I don't speak that language" ,

 . The Relationship: Architecture vs. Compute Capability

    Architecture (Ampere) : This is the Generation, it defines the broad "How-To" manual and it says : "All Ampere cards will have 3rd Gen Tensor Cores and 2nd Gen RT Cores" , 
    Compute Capability (8.0 vs 8.6) : This is the Version Number of the instruction set,and it tells you exactly what "extra features" or "limitations" the specific chip has , 

2. Why does the RTX 3090 have CC 8.6 while an A100 has CC 8.0 ? 
It sounds backwards, right? Usually, a higher number is better,but in the world of NVIDIA, these numbers often represent specialization , 
---> Compute capability 8.0(ex : NVIDIA A100) is used for Data center/AI/supercomputing applications with huge(164 KB) shared memory per SM , and with large L1 cache size,optimized for FP64(double-precision) math speed , whereas Compute-capability 8.6(ex : my  NVIDIA RTX 3060/3090) is specially targeted audience of gamers/creators/local workstations with smaller(100 KB) shared memory per SM and smaller L1-Cache size and optimized for FP32(single-precision) math speed , 

The Difference : 

    CC 8.0 (The Scientist) : The A100 GPU  is built for massive scientific simulations (like weather or physics) and it needs a massive "desk" (Shared Memory) to hold data while it works ,whereas 
    CC 8.6 (The Gamer/Worker) : my  RTX 3060 is built for speed , and it has a slightly "newer" instruction set (hence .6) that allows it to process twice as many FP32 operations per clock cycle compared to the older 8.0 design, but it has a smaller "desk" (Shared Memory) ,

3. What changes when CC changes (but Architecture stays the same) ? 
When the CC version moves from 8.0 to 8.6, the "Core Blueprint" stays the same, but three main things can change:

    Register Count / Shared Memory : The amount of fast, on-chip memory available to each thread might grow or shrink ,  this is the biggest headache for programmers because it changes how many "workers" (threads) we can fit on a chip at once ; 
    Throughput Ratios : An 8.6 card might be able to do 128 math operations per cycle, while an   8.0 card does 64, and  the "language" is the same, but the "speed" of certain math operations is tuned differently; 
    Specific Instruction Support : Sometimes, a .x update adds a tiny new feature ,  for example, CC 8.6 added specialized hardware support for "Asynchronous Copy" from global memory to shared memory , 

4. How does this affect OUR code ? 
If you are writing CUDA code : 

    If you compile for Compute 8.0, it will run perfectly on your 8.6 card (it’s "backwards compatible").
    If you write code specifically using a feature found only in 8.6, it will crash if you try to run it on an A100 (CC 8.0).


So,the 4-major designations of a GPU : 
1.Marketing Name ---> GeForce RTX 3060 --> (the Card or Board and this is what we buy at the store,it tells us the physical size and the number of fans) ;  
2.Silicon ID ---> GA106 ---> The GPU ( The Chip and a single "Card" might use different chip versions,and also engineers look at this to see exactly which "Engine" is inside )  ;  
3.Blueprint ---> Ampere ---> The Architecture class (This tells us the "DNA" of the card and it defines the layout of the cores and how they talk to memory and also tells about the generation of the card) ; 
4.Instruction Set ---> 8.6 ---> The Compute Capability (Crucial for coding,and this tells us exactly which cuda functions will work and which will crash); 

For our purposes , the primary focus is on the graphics card name/Model name/number(becoz that is the most available identifier when inspecting your system or shopping for new hardware) and compute capability (becoz that is what matters for coding and performance) . 


