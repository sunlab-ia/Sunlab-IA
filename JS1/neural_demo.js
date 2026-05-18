const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const WIDTH = canvas.width;
const HEIGHT = canvas.height;

const COLORS = {
    yellow: "#ffd400",
    red: "#ff3333",
    blue: "#2577e5",
    orange: "#ff8c00",
    dark: "#0f172a",
    text: "#19263b"
};

const BASE_COLORS = [
    {
        name: "JAUNE",
        code: [1,0,0],
        color: COLORS.yellow,
        target: [1,0,0]
    },
    {
        name: "ROUGE",
        code: [0,1,0],
        color: COLORS.red,
        target: [0,1,0]
    },
    {
        name: "BLEU",
        code: [0,0,1],
        color: COLORS.blue,
        target: [0,0,1]
    }
];

function dot(a,b){
    let s = 0;
    for(let i=0;i<a.length;i++){
        s += a[i]*b[i];
    }
    return s;
}

function softmax(values){
    const max = Math.max(...values);
    const exps = values.map(v => Math.exp(v-max));
    const sum = exps.reduce((a,b)=>a+b,0);
    return exps.map(v=>v/sum);
}

class SimpleNetwork {

    constructor(){
        this.w = [
            [0.5,0.5,0.5],
            [0.5,0.5,0.5],
            [0.5,0.5,0.5]
        ];

        this.b = [0,0,0];
    }

    forward(x){

        const raw = [];

        for(let o=0;o<3;o++){
            raw.push(dot(this.w[o],x)+this.b[o]);
        }

        return softmax(raw);
    }

    trainOne(x,target,lr=0.35){

        const probs = this.forward(x);

        const delta = [];

        for(let i=0;i<3;i++){
            delta.push(probs[i]-target[i]);
        }

        for(let o=0;o<3;o++){

            for(let i=0;i<3;i++){
                this.w[o][i] -= lr*delta[o]*x[i];
            }

            this.b[o] -= lr*delta[o];
        }
    }
}

class NeuralDemo {

    constructor(){

        this.net = new SimpleNetwork();

        this.trainingSet = [];

        for(let i=0;i<21;i++){
            this.trainingSet.push(BASE_COLORS[i%3]);
        }

        this.step = 0;

        this.current = BASE_COLORS[0];

        this.message = "Clique sur un bouton pour apprendre.";

        this.loop();
    }

    learnNext(){

        if(this.step >= this.trainingSet.length){
            this.message = "Apprentissage terminé.";
            return;
        }

        const ex = this.trainingSet[this.step];

        this.current = ex;

        this.net.trainOne(ex.code, ex.target);

        this.step++;

        this.message = `Exemple appris : ${ex.name}`;
    }

    learnAll(){

        while(this.step < this.trainingSet.length){
            this.learnNext();
        }
    }

    reset(){
        this.net = new SimpleNetwork();
        this.step = 0;
        this.current = BASE_COLORS[0];
        this.message = "Réseau réinitialisé.";
    }

    newColor(){

        const i = Math.floor(Math.random()*BASE_COLORS.length);
        this.current = BASE_COLORS[i];
    }

    drawNode(x,y,color,text){

        ctx.beginPath();
        ctx.arc(x,y,30,0,Math.PI*2);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.strokeStyle = "#111";
        ctx.lineWidth = 2;
        ctx.stroke();

        ctx.fillStyle = "black";
        ctx.font = "bold 14px Arial";
        ctx.textAlign = "center";
        ctx.fillText(text,x,y+50);
    }

    draw(){

        ctx.clearRect(0,0,WIDTH,HEIGHT);

        ctx.fillStyle = "#f6f8fc";
        ctx.fillRect(0,0,WIDTH,HEIGHT);

        ctx.fillStyle = "#111";
        ctx.font = "bold 32px Arial";
        ctx.fillText("Réseau de neurones pédagogique",40,60);

        ctx.font = "18px Arial";
        ctx.fillText(this.message,40,100);

        const inputs = [
            [450,300],
            [450,470],
            [450,640]
        ];

        const outputs = [
            [1000,300],
            [1000,470],
            [1000,640]
        ];

        const probs = this.net.forward(this.current.code);

        for(let i=0;i<3;i++){
            for(let o=0;o<3;o++){

                const w = this.net.w[o][i];

                ctx.strokeStyle = w > 0.5 ? COLORS.orange : "#999";
                ctx.lineWidth = Math.abs(w)*4;

                ctx.beginPath();
                ctx.moveTo(inputs[i][0]+30, inputs[i][1]);
                ctx.lineTo(outputs[o][0]-30, outputs[o][1]);
                ctx.stroke();
            }
        }

        this.drawNode(450,300,COLORS.yellow,"Jaune");
        this.drawNode(450,470,COLORS.red,"Rouge");
        this.drawNode(450,640,COLORS.blue,"Bleu");

        this.drawNode(1000,300,COLORS.yellow,probs[0].toFixed(2));
        this.drawNode(1000,470,COLORS.red,probs[1].toFixed(2));
        this.drawNode(1000,640,COLORS.blue,probs[2].toFixed(2));

        ctx.fillStyle = COLORS.text;
        ctx.font = "bold 22px Arial";
        ctx.fillText("Entrée actuelle : "+this.current.name, 560, 800);

        ctx.fillStyle = this.current.color;
        ctx.fillRect(920,760,120,120);
    }

    loop(){

        this.draw();

        requestAnimationFrame(()=>this.loop());
    }
}

const demo = new NeuralDemo();

document.getElementById("learnNext").onclick = ()=>demo.learnNext();
document.getElementById("learnAll").onclick = ()=>demo.learnAll();
document.getElementById("resetBtn").onclick = ()=>demo.reset();
document.getElementById("testBtn").onclick = ()=>demo.newColor();
